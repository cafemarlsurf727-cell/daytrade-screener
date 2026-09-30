#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
デイトレ候補スクリーナー v2 (GitHub Actions / Windows 対応、無料データ版)

初回: pip install pandas numpy yfinance requests exchange_calendars openpyxl xlrd
通常: python daytrade_screener.py --dry-run
200銘柄の構築: python daytrade_screener.py --build-universe --universe-batch-size 80
  JPXの一覧を公式ページから取得します（失敗時は --jpx-file 手動取得ファイル.xls）。
  毎回新規80銘柄ずつ過去1年を調査し、一定以上集まった時点で
  tickers_200.csv を自動生成。調査中は既存リストか内蔵サンプルを使用。
  universe/universe_metrics.csv をリポジトリに保存して続きから実行できます。
通常実行では tickers_200.csv が存在すれば自動使用します。

任意: --breadth-file market_breadth.csv
  外部市場騰落レシオCSV(date,advances,declines)を渡したときだけ過熱フィルター稼働。
任意: --shortable-file shortable.csv
  1列目に証券会社で当日空売り可能と確認した銘柄を指定。
  ファイル未指定時、C戦略は資金割当しません。毎朝の規制・在庫の再確認必須。
--dry-run は通知停止（ログやHTMLは生成するので本番前の検証で使用）。

注意: シグナルと翌日の検証は研究用の仮想結果です。
日足での損切り約定価格、寄り付き成行、貸株在庫は保証されません。
1取引日の最終判断・注文を15:25までに済ませ、持越しをしない想定です。
"""



from __future__ import annotations

import argparse
import io
import hashlib
import random
import re
from urllib.parse import urljoin
import dataclasses
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ============================================================================
# ロギング設定
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("daytrade_screener")

# Windows / GitHub Actions の両方で日本時間を使う。
JST = timezone(timedelta(hours=9))
JPX_LIST_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
MODEL_VERSION = "v2-open-slippage"
EXIT_GUIDE = "15:25までに手仕舞い判断・注文を完了（15:30の大引けに依存しない）"


def today_jst():
    return datetime.now(JST).date()


def previous_session(day=None):
    """今日より前に終了した直近の東証立会日。取引所カレンダーがあれば祝日も判定。"""
    day = day or today_jst()
    try:
        import exchange_calendars as xcals
        calendar = xcals.get_calendar("XTKS")
        for offset in range(1, 13):
            d = day - timedelta(days=offset)
            if calendar.is_session(pd.Timestamp(d)):
                return d
    except (ImportError, ValueError, KeyError) as exc:
        logger.warning("東証カレンダー未使用（平日近似）: %s", exc)
    for offset in range(1, 13):
        d = day - timedelta(days=offset)
        if d.weekday() < 5:
            return d
    raise ValueError("前営業日を特定できません")


def is_trading_day(day=None):
    day = day or today_jst()
    if day.weekday() >= 5:
        return False
    try:
        import exchange_calendars as xcals
        return bool(xcals.get_calendar("XTKS").is_session(pd.Timestamp(day)))
    except (ImportError, ValueError, KeyError) as exc:
        logger.warning("祝日判定にexchange_calendarsが必要です: %s", exc)
        return True  # 未導入なら平日のみ判定。GitHub Actions には導入推奨。


# ============================================================================
# デフォルト銘柄リスト
# ----------------------------------------------------------------------------
# 東証プライムの代表的な流動性の高い銘柄をサンプルとして同梱しています。
# 実運用では --tickers-file で「東証上場銘柄一覧（JPX公式CSV）」などから
# 生成した数百〜全銘柄リストを渡すことを強く推奨します。
# （このサンプルのままだと母集団が小さく、シグナル数が少なくなります）
# ============================================================================
DEFAULT_TICKERS = [
    "7203.T", "6758.T", "9984.T", "8306.T", "9432.T", "6861.T", "6501.T",
    "8035.T", "6098.T", "4063.T", "9433.T", "7267.T", "6367.T", "8058.T",
    "8031.T", "8001.T", "4568.T", "6902.T", "6503.T", "7741.T",
    "6273.T", "9983.T", "4661.T", "6954.T", "8766.T", "8316.T", "8411.T",
    "6981.T", "7013.T", "5108.T", "4507.T", "4519.T", "6178.T", "9020.T",
    "9022.T", "9101.T", "9104.T", "1605.T", "5401.T", "5713.T", "7269.T",
    "7270.T", "6301.T", "6326.T", "6752.T", "6723.T", "6971.T", "6920.T",
    "4755.T", "3382.T", "8267.T", "2914.T", "4502.T", "4503.T", "4523.T",
]


# ============================================================================
# 設定値（戦略パラメータ）
# ----------------------------------------------------------------------------
# ここを変えるだけで各戦略の閾値をチューニングできます。
# ============================================================================
@dataclasses.dataclass
class StrategyConfig:
    # Strategy A: 逆張り買い（5日乖離率 <= -5% かつ 前日が大陰線）
    a_dev5_threshold: float = -5.0
    a_bear_candle_ratio: float = 0.95  # 終値 <= 始値 * この比率

    # Strategy B: 押し目買い（3日乖離率が -10%〜-25%の範囲）
    b_dev3_lower: float = -25.0
    b_dev3_upper: float = -10.0

    # Strategy C: 急騰株空売り（前日終値比 +20%以上）
    c_pct_change_threshold: float = 20.0

    # Strategy D: 下髭サポート買い（下髭 > 終値×0.05 かつ 前日比 <= -8%）
    d_lower_shadow_ratio: float = 0.05
    d_pct_change_threshold: float = -8.0  # 「前日比 <= 0.92」＝-8%以下 と解釈

    # 地合いフィルター（騰落レシオ, %）
    # 一般に 120〜130 超で「買われすぎ」、70〜80 未満で「売られすぎ」とされる
    overheated_ad_ratio: float = 125.0
    oversold_ad_ratio: float = 70.0

    # リスク管理
    atr_period: int = 14
    # 戦略コードごとのATR倍率（損切り = エントリー ∓ ATR×この倍率）。
    # 単一の1.5倍だと全戦略で「狩られやすい＝損切り貧乏」になりがちなので、
    # 戦略の性質（踏み上げリスクの有無、既にボラが高まった後に入るか等）に
    # 応じて個別に調整できるようにしている。値はまだ未検証の初期値であり、
    # SignalLogger に溜まったデータを使って後日チューニングする前提。
    atr_multiplier_by_strategy: dict = dataclasses.field(
        default_factory=lambda: {
            "A": 1.5,   # 逆張り買い
            "B": 1.5,   # 押し目買い
            "C": 2.0,   # 急騰株空売り：踏み上げリスクがあるため広めに設定
            "D": 1.5,   # 下髭サポート買い
        }
    )
    # strategy_codeが未知の場合や後方互換のためのデフォルト倍率
    atr_multiplier_default: float = 1.5
    unit_shares: int = 100  # 日本株の単元株数（多くは100株単位）


CONFIG = StrategyConfig()


# ============================================================================
# 1. DataLoader：無料枠を意識した分割取得＋増分キャッシュ
# ============================================================================
class DataLoader:
    def __init__(self, tickers, cache_dir="cache", period="1y", batch_size=25,
                 max_retries=2, retry_wait_sec=3.0):
        self.tickers = list(dict.fromkeys(tickers))
        self.cache_dir = Path(cache_dir) / "tickers"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.period = period  # yfinance の有効値: 1mo / 3mo / 6mo / 1y ...
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_wait_sec = retry_wait_sec
        self.expected_day = previous_session()
        self.failed = []

    def _cache_path(self, ticker):
        if not re.fullmatch(r"[A-Z0-9.\-]+", ticker):
            raise ValueError(f"不正なティッカー: {ticker}")
        return self.cache_dir / f"{ticker}.csv"

    @staticmethod
    def normalize(df):
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
        cols = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in out.columns for c in cols):
            return pd.DataFrame()
        out = out[cols]
        out.index = pd.DatetimeIndex(out.index)
        if out.index.tz is not None:
            out.index = out.index.tz_convert("Asia/Tokyo").tz_localize(None)
        out.index = out.index.normalize()
        for col in cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=cols[:4])
        out = out[(out["Close"] > 0) & (out["Volume"] >= 0)]
        out = out.loc[out.index.date < today_jst()]  # 当日未確定足を絶対に使わない
        return out.loc[~out.index.duplicated(keep="last")].sort_index().tail(320)

    def _read_cache(self, ticker):
        path = self._cache_path(ticker)
        if not path.exists():
            return pd.DataFrame()
        try:
            return self.normalize(pd.read_csv(path, index_col=0, parse_dates=True))
        except Exception as exc:
            logger.warning("キャッシュ破損 %s: %s", ticker, exc)
            return pd.DataFrame()

    def _save_cache(self, ticker, df):
        if not df.empty:
            df.to_csv(self._cache_path(ticker), index_label="Date")

    @staticmethod
    def _unpack(batch_data, ticker, batch_count):
        if batch_data is None or batch_data.empty:
            return pd.DataFrame()
        if isinstance(batch_data.columns, pd.MultiIndex):
            if ticker in batch_data.columns.get_level_values(0):
                return batch_data[ticker]
            if ticker in batch_data.columns.get_level_values(1):
                return batch_data.xs(ticker, axis=1, level=1)
            return pd.DataFrame()
        return batch_data if batch_count == 1 else pd.DataFrame()

    def _download(self, tickers, *, period=None, start=None):
        params = dict(tickers=tickers, interval="1d", group_by="ticker",
                      auto_adjust=True, progress=False, threads=False, timeout=20)
        if start is None:
            params["period"] = period or self.period  # 期間指定時はendと併用しない
        else:
            params["start"] = start
            params["end"] = today_jst().isoformat()  # endは排他的
        return yf.download(**params)

    def _collect_batches(self, tickers, result, *, incremental=False):
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            logger.info("株価取得 %d-%d / %d 銘柄", i + 1, i + len(batch), len(tickers))
            start = None
            if incremental:
                start = min((result[t].index[-1] - pd.Timedelta(days=6)).date()
                            for t in batch).isoformat()
            fresh = None
            for attempt in range(self.max_retries):
                try:
                    fresh = self._download(batch, start=start)
                    if fresh is not None and not fresh.empty:
                        break
                except Exception as exc:
                    logger.warning("一括取得失敗 (%s件) : %s", len(batch), exc)
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_wait_sec * (attempt + 1))
            for ticker in batch:
                raw = self._unpack(fresh, ticker, len(batch))
                normalized = self.normalize(raw)
                if normalized.empty:
                    logger.warning("株価未取得: %s（キャッシュがあれば保持）", ticker)
                    continue
                old = result.get(ticker, pd.DataFrame())
                merged = self.normalize(pd.concat([old, normalized])) if not old.empty else normalized
                if len(merged) >= 15:
                    result[ticker] = merged
                    self._save_cache(ticker, merged)
            failed_batch = [t for t in batch if t not in result or
                            result[t].index[-1].date() < self.expected_day]
            # 1-4銘柄だけ欠損なら個別で再試行。大量欠損時はレート制限を疑い連打しない。
            if 0 < len(failed_batch) <= 4:
                for ticker in failed_batch:
                    try:
                        time.sleep(self.retry_wait_sec)
                        extra = self._download([ticker], start=start)
                        normalized = self.normalize(self._unpack(extra, ticker, 1))
                        if not normalized.empty:
                            old = result.get(ticker, pd.DataFrame())
                            merged = self.normalize(pd.concat([old, normalized])) if not old.empty else normalized
                            if len(merged) >= 15:
                                result[ticker] = merged
                                self._save_cache(ticker, merged)
                    except Exception as exc:
                        logger.warning("個別再取得失敗 %s: %s", ticker, exc)
            if i + self.batch_size < len(tickers):
                time.sleep(0.7)

    def fetch_all(self):
        result = {}
        missing, update = [], []
        for ticker in self.tickers:
            cached = self._read_cache(ticker)
            if cached.empty:
                missing.append(ticker)
            else:
                result[ticker] = cached
                if cached.index[-1].date() < self.expected_day:
                    update.append(ticker)
        if missing:
            self._collect_batches(missing, result, incremental=False)
        if update:
            self._collect_batches(update, result, incremental=True)
        # 空や古いデータを利用して前々日のシグナルを再通知する事故を防ぐ。
        valid = {t: df for t, df in result.items()
                 if len(df) >= 15 and df.index[-1].date() == self.expected_day}
        self.failed = [t for t in self.tickers if t not in valid]
        logger.info("株価データ %d/%d 成功、取得失敗・日付不一致 %d",
                    len(valid), len(self.tickers), len(self.failed))
        if self.failed:
            logger.warning("未取得の先頭20件: %s", ", ".join(self.failed[:20]))
        return valid


# ============================================================================
# 2. TechnicalEngine モジュール
# ============================================================================
class TechnicalEngine:
    """
    各種テクニカル指標をベクトル演算（pandas）で計算するクラス。
    """

    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        1銘柄分のOHLCVデータフレームに、以下の列を追加して返す。

        - ma3, ma5, ma10          : 単純移動平均
        - dev3, dev5, dev10       : 終値の移動平均乖離率(%)
        - roc9                    : 9日ROC（Rate of Change, %）
        - pct_change              : 前日比騰落率(%)
        - lower_shadow_ratio      : 下髭の長さ ÷ 終値
        - is_bear_candle          : 大陰線判定（終値 <= 始値×閾値）用の比率
        - atr14                   : ATR（真の値幅の指数移動平均）
        """
        out = df.copy()
        close = out["Close"]
        high = out["High"]
        low = out["Low"]
        open_ = out["Open"]

        # --- 移動平均・乖離率 ---
        for window in (3, 5, 10):
            ma = close.rolling(window=window, min_periods=window).mean()
            out[f"ma{window}"] = ma
            out[f"dev{window}"] = (close - ma) / ma * 100.0

        # --- ROC(9日) ---
        out["roc9"] = close.pct_change(periods=9) * 100.0

        # --- 前日比騰落率 ---
        out["pct_change"] = close.pct_change(periods=1) * 100.0

        # --- 下髭の長さ比率（ローソク足の実体下端 or 終値・始値の低い方 - 安値）---
        body_bottom = pd.concat([open_, close], axis=1).min(axis=1)
        lower_shadow = (body_bottom - low).clip(lower=0)
        out["lower_shadow_ratio"] = lower_shadow / close

        # --- 始値に対する終値の比率（大陰線判定用） ---
        out["close_open_ratio"] = close / open_

        # --- ATR（Average True Range） ---
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr14"] = tr.rolling(window=CONFIG.atr_period, min_periods=CONFIG.atr_period).mean()

        return out

    @staticmethod
    def compute_market_breadth(data_dict, breadth_file=None, as_of=None):
        """対象銘柄の平均乖離率と、独立した外部CSVがある場合のみ市場騰落レシオを返す。"""
        dev5 = []
        for df in data_dict.values():
            if len(df) >= 5:
                x = TechnicalEngine.compute_indicators(df)["dev5"].iloc[-1]
                if pd.notna(x):
                    dev5.append(float(x))
        breadth = {"ad_ratio": None,
                   "avg_dev5": float(np.mean(dev5)) if dev5 else 0.0,
                   "ad_source": "未設定（市場フィルターは無効）"}
        if not breadth_file:
            return breadth
        try:
            external = pd.read_csv(breadth_file)
            required = {"date", "advances", "declines"}
            if not required.issubset(external.columns):
                raise ValueError(f"CSVに {required} の列が必要です")
            external["date"] = pd.to_datetime(external["date"], errors="coerce")
            external["advances"] = pd.to_numeric(external["advances"], errors="coerce")
            external["declines"] = pd.to_numeric(external["declines"], errors="coerce")
            cutoff = pd.Timestamp(as_of or previous_session())
            external = external.dropna(subset=["date", "advances", "declines"])
            external = external[external["date"] <= cutoff].sort_values("date").drop_duplicates("date")
            if len(external) < 25 or external.iloc[-1]["date"].date() != cutoff.date():
                raise ValueError("25営業日以上のデータ、直近立会日の値が必要です")
            last25 = external.tail(25)
            decline = float(last25["declines"].sum())
            advance = float(last25["advances"].sum())
            if decline == 0:
                raise ValueError("25日間の値下がり銘柄数合計がゼロです")
            breadth.update(ad_ratio=advance / decline * 100.0,
                           ad_source=f"外部CSV: {breadth_file}")
        except Exception as exc:
            logger.warning("騰落レシオを無効化しました: %s", exc)
        return breadth


# ============================================================================
# 3. SignalScreener モジュール
# ============================================================================
class SignalScreener:
    """
    各銘柄のテクニカル指標から、4種類の売買シグナルを判定するクラス。
    地合いフィルターによるシグナル抑制は RiskEngine 側と連携して行う。
    """

    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config

    def _latest_row(self, indicators: pd.DataFrame) -> Optional[pd.Series]:
        if indicators.empty:
            return None
        row = indicators.iloc[-1]
        # 主要指標が計算できていない（データ不足）銘柄は除外
        required = ["dev3", "dev5", "pct_change", "atr14", "lower_shadow_ratio"]
        if row[required].isna().any():
            return None
        return row

    def _check_strategy_a(self, row: pd.Series) -> bool:
        """逆張り買い: 5日乖離率 <= -5% かつ 前日が大陰線（終値<=始値×0.95）"""
        return (
            row["dev5"] <= self.config.a_dev5_threshold
            and row["close_open_ratio"] <= self.config.a_bear_candle_ratio
        )

    def _check_strategy_b(self, row: pd.Series) -> bool:
        """押し目買い: 3日乖離率が -10%〜-25%の範囲内"""
        return self.config.b_dev3_lower <= row["dev3"] <= self.config.b_dev3_upper

    def _check_strategy_c(self, row: pd.Series) -> bool:
        """急騰株空売り: 前日終値比 +20%以上"""
        return row["pct_change"] >= self.config.c_pct_change_threshold

    def _check_strategy_d(self, row: pd.Series) -> bool:
        """下髭サポート買い: 下髭 > 終値×0.05 かつ 前日比 <= -8%"""
        return (
            row["lower_shadow_ratio"] > self.config.d_lower_shadow_ratio
            and row["pct_change"] <= self.config.d_pct_change_threshold
        )

    def screen(self, data_dict, names=None, suppress_reverse_buy=False):
        """1銘柄につき1件。重複条件はmatched_strategiesに保持して二重投資しない。"""
        signals = []
        names = names or {}
        priority = {"D": 0, "A": 1, "B": 2, "C": 3}
        for ticker, df in data_dict.items():
            ind = TechnicalEngine.compute_indicators(df)
            row = self._latest_row(ind)
            if row is None or float(row["Volume"]) <= 0:
                continue
            matches = []
            if not suppress_reverse_buy and self._check_strategy_a(row):
                matches.append(("A", "逆張り買い", "buy"))
            if not suppress_reverse_buy and self._check_strategy_b(row):
                matches.append(("B", "急落後の逆張り買い", "buy"))
            if self._check_strategy_c(row):
                matches.append(("C", "急騰株空売り・貸株要確認", "sell"))
            if not suppress_reverse_buy and self._check_strategy_d(row):
                matches.append(("D", "下髭サポート買い", "buy"))
            if not matches:
                continue
            matches.sort(key=lambda x: priority[x[0]])
            code, label, side = matches[0]
            close = float(row["Close"])
            atr = float(row["atr14"])
            turnover20 = float((df["Close"] * df["Volume"]).tail(20).mean())
            signal = {"ticker": ticker, "name": names.get(ticker, ticker),
                      "strategy_code": code, "strategy_label": label, "side": side,
                      "matched_strategies": "+".join(x[0] for x in matches),
                      "entry_price": close, "atr": atr,
                      "turnover20": turnover20,
                      "dev3": float(row["dev3"]), "dev5": float(row["dev5"]),
                      "pct_change": float(row["pct_change"]),
                      "lower_shadow_ratio": float(row["lower_shadow_ratio"]),
                      "shortable_verified": False}
            # 単純な極端値の強さで表示順だけ決める。期待収益の予測値ではない。
            if code == "A":
                signal["severity"] = max(0.0, -signal["dev5"] / 5)
            elif code == "B":
                signal["severity"] = max(0.0, -signal["dev3"] / 10)
            elif code == "C":
                signal["severity"] = max(0.0, signal["pct_change"] / 20)
            else:
                signal["severity"] = max(0.0, -signal["pct_change"] / 8)
            signals.append(signal)
        signals.sort(key=lambda x: (-x["severity"], -x["turnover20"], x["ticker"]))
        logger.info("シグナル: %d銘柄（重複戦略を統合済み）", len(signals))
        return signals


# ============================================================================
# 4. NoviceGuardrail & RiskEngine モジュール
# ============================================================================
class RiskEngine:
    """
    初心者保護のためのガードレールと、資金管理計算を担当するクラス。

    - 地合いフィルター: 極端な買われすぎ相場では逆張り買い系シグナルを抑制する
    - ATRベースの動的損切り: ATR×戦略別倍率(StrategyConfig参照)を損切り目安とし、狭すぎる損切りに
      よる「ノイズでの不要な損切り（＝貧乏ゆすりカット）」を避ける
    - ポジションサイズ計算: 総資金と許容リスク(%)から最大購入可能株数を算出
    """

    def __init__(
        self,
        capital: float,
        risk_pct: float = 0.01,
        config: StrategyConfig = CONFIG,
    ) -> None:
        if capital <= 0:
            raise ValueError("capital は正の値である必要があります。")
        if not (0 < risk_pct <= 0.1):
            raise ValueError("risk_pct は 0〜0.1（0〜10%）の範囲を推奨します。")
        self.capital = capital
        self.risk_pct = risk_pct
        self.config = config

    def should_suppress_reverse_buy(self, breadth: dict[str, float]) -> bool:
        """
        市場全体が極端な買われすぎ状態かどうかを判定する。
        True の場合、逆張り買い系シグナル（A・B・D）を抑制すべき。
        """
        overheated = (breadth.get("ad_ratio") is not None
                      and breadth["ad_ratio"] >= self.config.overheated_ad_ratio)
        if overheated:
            logger.info(
                "地合いフィルター発動: 騰落レシオ %.1f%% >= %.1f%% のため、"
                "逆張り買い系シグナルを抑制します。",
                breadth["ad_ratio"], self.config.overheated_ad_ratio,
            )
        return overheated

    def calc_stop_loss(self, entry_price: float, atr: float, side: str, strategy_code: str) -> float:
        """
        ATR × 戦略別倍率 の位置に損切り価格を算出する。
        倍率は StrategyConfig.atr_multiplier_by_strategy から戦略コードで引く
        （未定義の戦略コードは atr_multiplier_default にフォールバック）。
        """
        multiplier = self.config.atr_multiplier_by_strategy.get(
            strategy_code, self.config.atr_multiplier_default
        )
        offset = atr * multiplier
        if side == "buy":
            stop = entry_price - offset
        else:  # sell（空売り）の場合は上側に損切りライン
            stop = entry_price + offset
        return round(max(stop, 0.0), 1)

    def calc_position_size(self, entry_price: float, stop_loss_price: float) -> dict:
        """
        1トレードあたりの許容リスク額（総資金×risk_pct）から、
        最大購入可能株数（単元株ベース）を算出する。
        """
        risk_amount = self.capital * self.risk_pct
        per_share_risk = abs(entry_price - stop_loss_price)

        if per_share_risk <= 0:
            return {
                "risk_amount": risk_amount,
                "per_share_risk": 0.0,
                "max_shares": 0,
                "max_lots": 0,
                "estimated_cost": 0.0,
                "warning": "損切り幅が0のためポジションサイズを計算できません。",
            }

        raw_shares = risk_amount / per_share_risk
        unit = self.config.unit_shares
        max_lots = int(raw_shares // unit)
        max_shares = max_lots * unit
        estimated_cost = max_shares * entry_price

        warning = None
        if max_shares == 0:
            warning = (
                "許容リスク額に対して損切り幅が広すぎるため、"
                f"最低単元（{unit}株）すら購入できません。見送り推奨。"
            )
        elif estimated_cost > self.capital:
            # 資金を超える場合は資金上限で再計算
            affordable_lots = int((self.capital // entry_price) // unit)
            max_shares = affordable_lots * unit
            estimated_cost = max_shares * entry_price
            warning = "リスク基準の株数が資金上限を超えるため、資金上限で調整しました。"

        return {
            "risk_amount": round(risk_amount, 0),
            "per_share_risk": round(per_share_risk, 2),
            "max_shares": max_shares,
            "max_lots": max_shares // unit if unit else 0,
            "estimated_cost": round(estimated_cost, 0),
            "warning": warning,
        }

    def enrich_signal(self, signal: dict) -> dict:
        """シグナル1件に、損切り価格・ポジションサイズ情報を付与する。"""
        multiplier = self.config.atr_multiplier_by_strategy.get(
            signal["strategy_code"], self.config.atr_multiplier_default
        )
        stop_loss = self.calc_stop_loss(
            signal["entry_price"], signal["atr"], signal["side"], signal["strategy_code"]
        )
        sizing = self.calc_position_size(signal["entry_price"], stop_loss)
        signal = dict(signal)
        signal["stop_loss"] = stop_loss
        signal["atr_multiplier"] = multiplier
        signal.update(sizing)
        return signal


    def allocate_portfolio(self, enriched, max_positions=3, portfolio_risk_pct=0.03,
                           shortable_tickers=None):
        """単元・合計現金・合計想定損失・最大ポジション数を同時に守る。空売りは別途貸株確認必須。"""
        remaining_cash = self.capital
        remaining_risk = self.capital * portfolio_risk_pct
        selected = []
        shortable_tickers = shortable_tickers or set()
        seen_tickers = set()
        for s in enriched:
            if s["ticker"] in seen_tickers:
                continue
            if len(selected) >= max_positions:
                break
            if s["side"] == "sell" and s["ticker"] not in shortable_tickers:
                logger.warning("空売り在庫未確認につき研究候補のみ（割当対象外）: %s", s["ticker"])
                continue
            per_share_risk = abs(s["entry_price"] - s["stop_loss"])
            if per_share_risk <= 0:
                continue
            unit = self.config.unit_shares
            shares_by_risk = int(min(self.capital * self.risk_pct, remaining_risk)
                                 // (per_share_risk * unit)) * unit
            # 空売りも保守的に売建想定元本を現金余力から差し引く
            shares_by_cash = int(remaining_cash // (s["entry_price"] * unit)) * unit
            shares = min(shares_by_risk, shares_by_cash)
            if shares < unit:
                continue
            item = dict(s)
            item.update(max_shares=shares, max_lots=shares // unit,
                        estimated_cost=round(shares * s["entry_price"], 0),
                        risk_amount=round(shares * per_share_risk, 0),
                        planned_risk_amount=round(self.capital * self.risk_pct, 0))
            if item["side"] == "sell":
                item["shortable_verified"] = False  # リスト入り≠発注時の実際の在庫・規制
                item["warning"] = "空売り候補：必ず当日発注前に証券会社の在庫・規制を再確認"
            item["warning"] = ((item.get("warning") or "") +
                               " / 寄り付きの値飛びで参考株数は変わります").strip(" / ")
            selected.append(item)
            seen_tickers.add(s["ticker"])
            remaining_cash -= shares * s["entry_price"]
            remaining_risk -= shares * per_share_risk
        logger.info("資金配分: %d件、現金予定%.0f円、合計想定リスク%.0f円",
                    len(selected), self.capital - remaining_cash,
                    self.capital * portfolio_risk_pct - remaining_risk)
        return selected


# ============================================================================
# 5. SignalLogger モジュール（検証運用：シグナルの記録と答え合わせ）
# ----------------------------------------------------------------------------
# 「シグナルを出す」だけでなく「そのシグナルが実際どうなったか」を
# 自動で記録・答え合わせするための仕組み。
#
# 記録タイミング：
#   当日朝、シグナルを検出した時点で status=pending として1行追加する。
#   このとき entry_price は「前日終値（シグナル判定の基準値）」であり、
#   実際の寄り付き価格とは異なる点に注意（後述の gap で乖離を可視化する）。
#
# 答え合わせタイミング：
#   翌営業日以降の実行時、DataLoaderが取得したその後の日足データの中に
#   signal_date の実際の始値・高値・安値・終値が含まれるようになるので、
#   それを使って損切りに触れたか／引けでいくらだったかを判定し、
#   status=resolved に更新する。追加のAPI呼び出しは不要。
# ============================================================================
class SignalLogger:
    """朝の仮説を記録し、翌営業日以降、始値エントリーを仮定して日足で検証。"""
    _COLUMNS = [
        "signal_date", "as_of_date", "ticker", "name", "strategy_code",
        "matched_strategies", "strategy_label", "side", "entry_price", "stop_loss",
        "max_shares", "estimated_cost", "risk_amount", "ad_ratio", "model_version",
        "status", "actual_open", "actual_high", "actual_low", "actual_close",
        "modeled_entry", "modeled_shares", "stopped_out", "exit_price", "gap_pct",
        "pnl_per_share", "pnl_total", "r_multiple", "resolved_at", "skip_reason",
    ]

    def __init__(self, log_path="logs/signal_log.csv", max_gap_pct=5.0, slippage_pct=0.001):
        self.log_path = Path(log_path)
        self.max_gap_pct = max_gap_pct
        self.slippage_pct = slippage_pct

    def load(self):
        if not self.log_path.exists():
            return pd.DataFrame(columns=self._COLUMNS)
        try:
            df = pd.read_csv(self.log_path, dtype={"ticker": str})
            if "model_version" not in df.columns:
                df["model_version"] = "legacy"  # 旧ロジックの成績は混ぜない
            for c in self._COLUMNS:
                if c not in df.columns:
                    df[c] = pd.NA
            return df[self._COLUMNS]
        except Exception as exc:
            # 破損時に既存記録を空データで上書きしない
            raise RuntimeError(f"ログを読めません。ログを保護するため処理中止: {exc}") from exc

    def save(self, df):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.log_path.with_suffix(".tmp")
        df.to_csv(temp, index=False)
        temp.replace(self.log_path)
        logger.info("シグナルログ保存: %d件", len(df))

    def append_new(self, df, signals, breadth, signal_date, as_of_date=None):
        existing = set(zip(df["signal_date"].astype(str), df["ticker"].astype(str),
                           df["strategy_code"].astype(str), df["model_version"].astype(str)))
        new_rows = []
        for s in signals:
            key = (signal_date, s["ticker"], s["strategy_code"], MODEL_VERSION)
            if key in existing:
                continue
            row = {col: pd.NA for col in self._COLUMNS}
            row.update({
                "signal_date": signal_date, "as_of_date": str(as_of_date or previous_session()),
                "ticker": s["ticker"], "name": s["name"],
                "strategy_code": s["strategy_code"],
                "matched_strategies": s.get("matched_strategies", s["strategy_code"]),
                "strategy_label": s["strategy_label"], "side": s["side"],
                "entry_price": s["entry_price"], "stop_loss": s["stop_loss"],
                "max_shares": s["max_shares"], "estimated_cost": s["estimated_cost"],
                "risk_amount": s["risk_amount"], "ad_ratio": breadth.get("ad_ratio"),
                "model_version": MODEL_VERSION, "status": "pending",
            })
            new_rows.append(row)
            existing.add(key)
        if new_rows:
            if df.empty:
                df = pd.DataFrame(new_rows, columns=self._COLUMNS)
            else:
                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        logger.info("本日の新規ログ %d件（重複は登録せず）", len(new_rows))
        return df

    def resolve(self, df, data_dict):
        if df.empty:
            return df
        write_cols = ["actual_open", "actual_high", "actual_low", "actual_close",
                      "modeled_entry", "modeled_shares", "stopped_out", "exit_price",
                      "gap_pct", "pnl_per_share", "pnl_total", "r_multiple", "resolved_at",
                      "status", "skip_reason"]
        df = df.copy()
        for c in write_cols:
            df[c] = df[c].astype(object)
        count = 0
        for idx in df[df["status"] == "pending"].index:
            row = df.loc[idx]
            ticker, signal_date = str(row["ticker"]), str(row["signal_date"])[:10]
            if ticker not in data_dict or signal_date >= today_jst().isoformat():
                continue
            bars = data_dict[ticker]
            if not any(str(date)[:10] == signal_date for date in bars.index):
                continue
            bar = bars.loc[[str(d)[:10] == signal_date for d in bars.index]].iloc[0]
            op, high, low, close = (float(bar[c]) for c in ["Open", "High", "Low", "Close"])
            previous_close = float(row["entry_price"])
            stop = float(row["stop_loss"])
            side = str(row["side"])
            gap = (op / previous_close - 1) * 100
            for col, val in [("actual_open", op), ("actual_high", high),
                             ("actual_low", low), ("actual_close", close), ("gap_pct", gap)]:
                df.at[idx, col] = val
            # 古いログに上書きされたv1実績が混入しないようにする。
            if row["model_version"] != MODEL_VERSION:
                df.at[idx, "status"] = "legacy_pending"
                df.at[idx, "skip_reason"] = "旧ロジック。新方式の成績から除外"
                continue
            skip_reason = None
            if abs(gap) > self.max_gap_pct:
                skip_reason = f"寄り付きギャップ{gap:+.2f}%が上限超過"
            elif (side == "buy" and op <= stop) or (side == "sell" and op >= stop):
                skip_reason = "寄り付き時点で事前想定の損切りラインを超過"
            elif op <= 0 or stop <= 0:
                skip_reason = "異常な価格"
            cost_budget = float(row["estimated_cost"])
            shares = min(int(row["max_shares"]), int(cost_budget // (op * CONFIG.unit_shares))
                         * CONFIG.unit_shares) if op > 0 else 0
            if shares == 0:
                skip_reason = skip_reason or "寄り付き時点の単元買付余力不足"
            if skip_reason:
                df.at[idx, "status"] = "skipped_gap"
                df.at[idx, "skip_reason"] = skip_reason
                df.at[idx, "resolved_at"] = datetime.now(JST).isoformat(timespec="seconds")
                count += 1
                continue
            stopped = low <= stop if side == "buy" else high >= stop
            theoretical_exit = stop if stopped else close
            slip = self.slippage_pct
            modeled_entry = op * (1 + slip if side == "buy" else 1 - slip)
            modeled_exit = theoretical_exit * (1 - slip if side == "buy" else 1 + slip)
            pnl_per_share = ((modeled_exit - modeled_entry) if side == "buy"
                             else (modeled_entry - modeled_exit))
            risk_per_share = abs(op - stop)
            result = {
                "modeled_entry": modeled_entry, "modeled_shares": shares,
                "stopped_out": bool(stopped), "exit_price": modeled_exit,
                "pnl_per_share": pnl_per_share, "pnl_total": shares * pnl_per_share,
                "r_multiple": pnl_per_share / risk_per_share if risk_per_share > 0 else pd.NA,
                "resolved_at": datetime.now(JST).isoformat(timespec="seconds"),
                "status": "resolved",
            }
            for col, val in result.items():
                df.at[idx, col] = val
            count += 1
        logger.info("仮想トレード答え合わせ %d件", count)
        return df

    @staticmethod
    def summarize(df):
        if df.empty:
            return {"overall": None, "by_strategy": {}}
        resolved = df[(df["status"] == "resolved") &
                      (df["model_version"] == MODEL_VERSION)].copy()
        if resolved.empty:
            return {"overall": None, "by_strategy": {}}
        resolved["r_multiple"] = pd.to_numeric(resolved["r_multiple"], errors="coerce")
        resolved = resolved.dropna(subset=["r_multiple"])
        def agg(sub):
            return {"n": len(sub), "win_rate": float((sub["r_multiple"] > 0).mean() * 100),
                    "avg_r": float(sub["r_multiple"].mean())}
        if resolved.empty:
            return {"overall": None, "by_strategy": {}}
        return {"overall": agg(resolved), "by_strategy": {
            code: agg(sub) for code, sub in resolved.groupby("strategy_code")}}


# ============================================================================
# 6. Notification モジュール
# ============================================================================
class Notifier:
    """
    スクリーニング結果を整形し、Discord Webhook / LINE Messaging API に
    送信するクラス。
    """

    def __init__(
        self,
        discord_webhook_url: Optional[str] = None,
        line_channel_access_token: Optional[str] = None,
        line_user_id: Optional[str] = None,
    ) -> None:
        self.discord_webhook_url = discord_webhook_url
        self.line_channel_access_token = line_channel_access_token
        self.line_user_id = line_user_id

    @staticmethod
    def format_message(signals: list[dict], breadth: dict[str, float]) -> str:
        """通知用のテキストメッセージを組み立てる。"""
        today_str = datetime.now(JST).strftime("%Y-%m-%d")
        lines = [
            f"【デイトレ候補銘柄】{today_str}",
            f"外部市場騰落レシオ: {breadth['ad_ratio']:.1f}% / "
            f"対象銘柄平均5日乖離率 {breadth['avg_dev5']:.2f}%"
            if breadth.get("ad_ratio") is not None else
            f"市場騰落レシオ: 未取得（抑制無効） / 対象平均5日乖離率 {breadth['avg_dev5']:.2f}%",
            f"※参考候補です。{EXIT_GUIDE}。持ち越ししない前提。",
            "",
        ]

        if not signals:
            lines.append("本日は条件を満たす候補銘柄がありませんでした。無理に売買しないこと。")
            return "\n".join(lines)

        for s in signals:
            side_label = "買い" if s["side"] == "buy" else "空売り"
            lines.append(
                f"■ {s['name']} ({s['ticker']}) [{s['strategy_label']} / {side_label}]\n"
                f"  前日終値（判定基準）: {s['entry_price']:.1f}円\n"
                f"  損切り目安    : {s['stop_loss']:.1f}円 (ATR×{s.get('atr_multiplier', 1.5):.1f})\n"
                f"  参考株数      : {s['max_shares']}株"
                f"（概算コスト {s['estimated_cost']:,.0f}円）\n"
                f"  想定リスク額  : {s['risk_amount']:,.0f}円\n"
            )
            if s.get("warning"):
                lines.append(f"  ⚠ {s['warning']}\n")

        lines.append(EXIT_GUIDE + "。寄り付き価格で参考株数は変わります。")
        return "\n".join(lines)

    def send_discord(self, message: str) -> bool:
        if not self.discord_webhook_url:
            logger.info("DISCORD_WEBHOOK_URL 未設定のため Discord 通知はスキップします。")
            return False
        try:
            # Discordは1メッセージ2000文字制限があるため分割送信する
            chunks = [message[i:i + 1900] for i in range(0, len(message), 1900)] or [message]
            for chunk in chunks:
                resp = requests.post(
                    self.discord_webhook_url,
                    json={"content": chunk},
                    timeout=10,
                )
                resp.raise_for_status()
            logger.info("Discord通知を送信しました。")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Discord通知の送信に失敗しました: %s", exc)
            return False

    def send_line(self, message: str) -> bool:
        if not (self.line_channel_access_token and self.line_user_id):
            logger.info(
                "LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID 未設定のため "
                "LINE通知はスキップします。"
            )
            return False
        try:
            # LINE Messaging APIは1メッセージ5000文字制限
            text = message[:4900]
            resp = requests.post(
                "https://api.line.me/v2/bot/message/push",
                headers={
                    "Authorization": f"Bearer {self.line_channel_access_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "to": self.line_user_id,
                    "messages": [{"type": "text", "text": text}],
                },
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("LINE通知を送信しました。")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("LINE通知の送信に失敗しました: %s", exc)
            return False

    @staticmethod
    def format_short_message(
        signals: list[dict], breadth: dict[str, float], pages_url: Optional[str] = None
    ) -> str:
        """
        LINE等向けの短い通知文。詳細はレポートページ側に任せ、
        通知はスマホ通知欄で一目で分かる要約に絞る。
        """
        today_str = datetime.now(JST).strftime("%Y-%m-%d")
        buy_count = sum(1 for s in signals if s["side"] == "buy")
        sell_count = sum(1 for s in signals if s["side"] == "sell")

        lines = [f"【デイトレ候補】{today_str}"]
        if signals:
            lines.append(f"検出 {len(signals)}件（買い{buy_count} / 空売り{sell_count}）")
        else:
            lines.append("本日は該当銘柄なし。無理に売買しないこと。")
        lines.append(f"市場騰落レシオ: {breadth['ad_ratio']:.1f}%" if breadth.get("ad_ratio") is not None else "市場騰落レシオ: 未設定")
        lines.append("※15:25までの手仕舞いを想定。")
        if pages_url:
            lines.append(f"詳細: {pages_url}")
        return "\n".join(lines)

    def notify(
        self,
        signals: list[dict],
        breadth: dict[str, float],
        dry_run: bool = False,
        pages_url: Optional[str] = None,
    ) -> None:
        full_message = self.format_message(signals, breadth)
        print("\n" + "=" * 70)
        print(full_message)
        print("=" * 70 + "\n")

        if dry_run:
            logger.info("--dry-run 指定のため外部通知は送信しません。")
            return

        # Discordは詳細をそのまま送る。LINEはレポートページへの導線として
        # 短い要約＋リンクのみ送る（pages_url未指定時はDiscordと同じ全文）。
        self.send_discord(full_message)
        short_message = (
            self.format_short_message(signals, breadth, pages_url=pages_url)
            if pages_url
            else full_message
        )
        self.send_line(short_message)


# ============================================================================
# 7. ReportGenerator モジュール（GitHub Pages 用レポートHTML生成）
# ----------------------------------------------------------------------------
# 新高値ブレイクツールと同系統の「サイバーパンク調」デザインで、
# メインカラーをシアン（水色）主体にした単一HTMLファイルを生成する。
# 外部通信は Google Fonts のみで、それ以外は完全に自己完結している。
# ============================================================================
class ReportGenerator:
    """スクリーニング結果を、GitHub Pagesで公開できる単一HTMLに整形するクラス。"""

    _STRATEGY_BADGE = {
        "A": ("逆張り買い", "buy"),
        "B": ("押し目買い", "buy"),
        "C": ("急騰株空売り", "sell"),
        "D": ("下髭サポート買い", "buy"),
    }

    @staticmethod
    def _escape(text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _render_card(self, s: dict) -> str:
        side_label = "BUY / 買い" if s["side"] == "buy" else "SHORT / 空売り"
        side_class = "side-buy" if s["side"] == "buy" else "side-sell"
        warning_html = (
            f'<p class="card-warning">⚠ {self._escape(s["warning"])}</p>'
            if s.get("warning")
            else ""
        )
        return f"""
        <article class="card">
          <div class="card-head">
            <span class="ticker">{self._escape(s['ticker'])}</span>
            <span class="side-badge {side_class}">{side_label}</span>
          </div>
          <h3 class="name">{self._escape(s['name'])}</h3>
          <p class="strategy">戦略: {self._escape(s['strategy_label'])} (Strategy {self._escape(s['strategy_code'])})</p>
          <dl class="metrics">
            <div><dt>前日終値（判定基準）</dt><dd>{s['entry_price']:.1f} 円</dd></div>
            <div><dt>損切り目安 (ATR×{s.get('atr_multiplier', 1.5):.1f})</dt><dd>{s['stop_loss']:.1f} 円</dd></div>
            <div><dt>参考株数（事前想定）</dt><dd>{s['max_shares']:,} 株</dd></div>
            <div><dt>概算コスト</dt><dd>{s['estimated_cost']:,.0f} 円</dd></div>
            <div><dt>想定リスク額</dt><dd>{s['risk_amount']:,.0f} 円</dd></div>
            <div><dt>前日比</dt><dd>{s['pct_change']:+.2f} %</dd></div>
          </dl>
          {warning_html}
        </article>
        """

    def _render_stats_panel(self, stats: Optional[dict]) -> str:
        if not stats or not stats.get("overall"):
            return """
        <section class="breadth-panel">
          <p class="breadth-label">検証実績（答え合わせ済みシグナル）</p>
          <p class="empty-state" style="padding:12px;">
            まだ答え合わせ済みのシグナルがありません。運用を続けると翌営業日以降、
            自動的にここに勝率・平均R倍数が蓄積されます。
          </p>
        </section>
        """
        overall = stats["overall"]
        rows = ""
        strategy_names = {"A": "逆張り買い", "B": "押し目買い", "C": "急騰空売り", "D": "下髭サポート"}
        for code in ["A", "B", "C", "D"]:
            s = stats["by_strategy"].get(code)
            if not s:
                continue
            rows += f"""
            <div class="stat-row">
              <span class="stat-label">{code}: {strategy_names.get(code, code)}</span>
              <span class="stat-value">勝率 {s['win_rate']:.0f}% / 平均R {s['avg_r']:+.2f} / n={s['n']}</span>
            </div>"""
        return f"""
        <section class="breadth-panel">
          <div class="breadth-row">
            <span class="breadth-label">検証実績（答え合わせ済み全体）</span>
            <span class="breadth-value">
              仮想勝率 {overall['win_rate']:.0f}% / 平均R {overall['avg_r']:+.2f} / n={overall['n']}
            </span>
          </div>
          <div class="stat-rows">{rows}</div>
          <p class="breadth-label" style="margin-top:8px;">
            ※ Rは1回あたりの想定リスク額に対する損益倍率。プラスが多いほど良い。
          </p>
        </section>
        """

    def build_html(
        self,
        signals: list[dict],
        breadth: dict[str, float],
        capital: float,
        risk_pct: float,
        stats: Optional[dict] = None,
    ) -> str:
        today_str = datetime.now(JST).strftime("%Y-%m-%d (%a)")
        cards_html = "\n".join(self._render_card(s) for s in signals)
        if not signals:
            cards_html = (
                '<p class="empty-state">本日は条件を満たす候補銘柄がありません。'
                "無理に売買しないこと。</p>"
            )

        stats_html = self._render_stats_panel(stats)

        ad_ratio = breadth.get("ad_ratio")
        # 騰落レシオを 0-200% のゲージにマッピング（100%を中央基準に）
        gauge_pct = max(0.0, min(100.0, (ad_ratio / 200.0) * 100.0)) if ad_ratio is not None else 0.0
        overheated = ad_ratio is not None and ad_ratio >= CONFIG.overheated_ad_ratio

        ad_display = f"{ad_ratio:.1f}%" if ad_ratio is not None else "未設定・判定無効"
        return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>デイトレ候補レポート | {today_str}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
  :root {{
    --accent: #22e6ff;
    --accent-soft: rgba(34, 230, 255, 0.07);
    --accent-magenta: #ff3ec8;
    --bg: #020305;
    --panel: #060a12;
    --panel-border: rgba(34, 230, 255, 0.15);
    --border: rgba(34, 230, 255, 0.18);
    --text: #c9edf5;
    --text-dim: #5d7d87;
    --danger: #ff4d6d;
    --warn: #ffcc33;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: 'Share Tech Mono', 'Courier New', monospace;
    padding-top: env(safe-area-inset-top, 0px);
    padding-bottom: env(safe-area-inset-bottom, 0px);
  }}
  body {{
    background-image:
      radial-gradient(circle at 15% 8%, var(--accent-soft), transparent 38%),
      radial-gradient(circle at 88% 0%, rgba(255,62,200,0.035), transparent 32%);
  }}
  .wrap {{ max-width: 1000px; margin: 0 auto; padding: 24px 16px 60px; }}
  header {{ margin-bottom: 20px; }}
  .title {{
    font-size: 1.6rem;
    letter-spacing: 0.08em;
    color: var(--accent);
    text-shadow: 0 0 6px rgba(34,230,255,0.35);
    margin: 0 0 4px;
  }}
  .date {{ color: var(--text-dim); margin: 0; }}
  .warning-banner {{
    border: 1px solid var(--danger);
    background: rgba(255, 77, 109, 0.08);
    color: var(--danger);
    padding: 10px 14px;
    border-radius: 6px;
    margin: 16px 0 24px;
    font-weight: bold;
  }}
  .breadth-panel {{
    border: 1px solid var(--panel-border);
    background: var(--panel);
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 24px;
  }}
  .breadth-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .breadth-label {{ color: var(--text-dim); font-size: 0.85rem; }}
  .breadth-value {{ color: var(--accent); font-size: 1.1rem; }}
  .gauge {{
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    margin-top: 10px;
    overflow: hidden;
  }}
  .gauge-fill {{
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent-magenta));
    width: {gauge_pct:.1f}%;
  }}
  .overheated-tag {{
    display: inline-block;
    margin-top: 8px;
    color: var(--warn);
    border: 1px solid var(--warn);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
  }}
  .stat-rows {{
    margin-top: 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .stat-row {{
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
    font-size: 0.82rem;
    border-top: 1px dashed var(--panel-border);
    padding-top: 4px;
  }}
  .stat-label {{ color: var(--text-dim); }}
  .stat-value {{ color: var(--accent); }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 14px;
  }}
  .card {{
    border: 1px solid var(--panel-border);
    background: var(--panel);
    border-radius: 10px;
    padding: 16px;
    transition: border-color .2s;
  }}
  .card:hover {{ border-color: var(--accent); }}
  .card-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }}
  .ticker {{ color: var(--accent); font-weight: bold; letter-spacing: 0.05em; }}
  .side-badge {{
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid currentColor;
  }}
  .side-buy {{ color: var(--accent); }}
  .side-sell {{ color: var(--accent-magenta); }}
  .name {{ margin: 4px 0; font-size: 1.02rem; color: var(--text); }}
  .strategy {{ color: var(--text-dim); font-size: 0.82rem; margin: 0 0 10px; }}
  .metrics {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 12px;
    margin: 0;
  }}
  .metrics dt {{ color: var(--text-dim); font-size: 0.72rem; }}
  .metrics dd {{ margin: 0; color: var(--text); font-size: 0.92rem; }}
  .card-warning {{
    margin: 10px 0 0;
    color: var(--warn);
    font-size: 0.78rem;
  }}
  .empty-state {{
    color: var(--text-dim);
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 24px;
    text-align: center;
  }}
  footer {{
    margin-top: 32px;
    color: var(--text-dim);
    font-size: 0.75rem;
    text-align: center;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg: #05070d; --text: #d9f9ff; }}
  }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <p class="title">DAYTRADE SIGNAL REPORT</p>
      <p class="date">{today_str}</p>
    </header>

    <div class="warning-banner">
      ⚠ 当日15:25までの手仕舞いを想定。成行・逆指値等の約定は保証されません。
    </div>

    <section class="breadth-panel">
      <div class="breadth-row">
        <span class="breadth-label">市場地合い（騰落レシオ）</span>
        <span class="breadth-value">{ad_display}</span>
      </div>
      <div class="gauge"><div class="gauge-fill"></div></div>
      <div class="breadth-row" style="margin-top:8px;">
        <span class="breadth-label">対象銘柄平均5日移動平均乖離率</span>
        <span class="breadth-value">{breadth['avg_dev5']:.2f}%</span>
      </div>
      {'<span class="overheated-tag">買われすぎ：逆張り買い系シグナルを抑制中</span>' if overheated else ''}
      <div class="breadth-row" style="margin-top:8px;">
        <span class="breadth-label">総資金 / 許容リスク</span>
        <span class="breadth-value">{capital:,.0f}円 / {risk_pct * 100:.1f}%</span>
      </div>
    </section>

    {stats_html}

    <section class="grid">
      {cards_html}
    </section>

    <footer>
      本レポートは仮想エントリーの研究用です。実際の貸株・注文・約定・諸費用は別途確認してください。<br>
      Generated by daytrade_screener.py
    </footer>
  </div>
</body>
</html>
"""

    def save(self, html: str, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        logger.info("レポートHTMLを出力しました: %s", out_path)


# ============================================================================
# 8. TradingSystem: 検証と新規シグナル。データ欠損時は誤通知しない。
# ============================================================================
class TradingSystem:
    def __init__(self, tickers, capital, risk_pct, cache_dir="cache", names=None,
                 report_path="docs/index.html", pages_url=None, log_path="logs/signal_log.csv",
                 breadth_file=None, shortable_file=None, max_positions=3,
                 portfolio_risk_pct=0.03, min_coverage=0.90,
                 max_gap_pct=5.0, slippage_pct=0.001):
        self.tickers = list(dict.fromkeys(tickers))
        self.names = names or {}
        self.capital = capital
        self.risk_pct = risk_pct
        self.report_path = report_path
        self.pages_url = pages_url
        self.breadth_file = breadth_file
        self.shortable_file = shortable_file
        self.max_positions = max_positions
        self.portfolio_risk_pct = portfolio_risk_pct
        self.min_coverage = min_coverage
        self.data_loader = DataLoader(tickers=self.tickers, cache_dir=cache_dir)
        self.screener = SignalScreener()
        self.risk_engine = RiskEngine(capital=capital, risk_pct=risk_pct)
        self.report_generator = ReportGenerator()
        self.signal_logger = (SignalLogger(log_path=log_path, max_gap_pct=max_gap_pct,
                                           slippage_pct=slippage_pct) if log_path else None)

    def run(self, dry_run=False):
        logger.info("=== デイトレ v2 開始、対象%d銘柄 ===", len(self.tickers))
        if not is_trading_day():
            logger.info("東証休場のため本日の新規通知はスキップします")
            return []
        existing_log = self.signal_logger.load() if self.signal_logger else None
        pending = []
        if existing_log is not None and not existing_log.empty:
            pending = existing_log.loc[existing_log["status"] == "pending", "ticker"].dropna().tolist()
        # 過去ログにしかいない銘柄も答え合わせ用に価格を再取得する。
        full_tickers = list(dict.fromkeys(self.tickers + pending))
        self.data_loader = DataLoader(full_tickers, cache_dir=str(self.data_loader.cache_dir.parent))
        all_data = self.data_loader.fetch_all()
        data = {t: all_data[t] for t in self.tickers if t in all_data}
        coverage = len(data) / len(self.tickers) if self.tickers else 0
        if coverage < self.min_coverage:
            logger.error("取得成功率 %.1f%% が基準 %.1f%% に未達。誤通知防止のため中止",
                         coverage * 100, self.min_coverage * 100)
            return []  # ログ上書きや'候補なし'通知はしない
        # 市場全体の騰落レシオは独立CSVが存在する場合のみ利用する。
        breadth = TechnicalEngine.compute_market_breadth(
            data, breadth_file=self.breadth_file, as_of=previous_session())
        suppress = self.risk_engine.should_suppress_reverse_buy(breadth)
        raw = self.screener.screen(data, names=self.names, suppress_reverse_buy=suppress)
        enriched = [self.risk_engine.enrich_signal(s) for s in raw]
        shortable = (set(load_tickers_from_csv(self.shortable_file))
                     if self.shortable_file and Path(self.shortable_file).exists() else set())
        actionable = self.risk_engine.allocate_portfolio(
            enriched, max_positions=self.max_positions,
            portfolio_risk_pct=self.portfolio_risk_pct, shortable_tickers=shortable)
        stats = None
        if self.signal_logger:
            log_df = self.signal_logger.resolve(existing_log, all_data)
            log_df = self.signal_logger.append_new(
                log_df, actionable, breadth, today_jst().isoformat(), previous_session())
            self.signal_logger.save(log_df)
            stats = self.signal_logger.summarize(log_df)
        if self.report_path:
            html = self.report_generator.build_html(
                actionable, breadth, capital=self.capital, risk_pct=self.risk_pct, stats=stats)
            self.report_generator.save(html, self.report_path)
        notifier = Notifier(
            discord_webhook_url=os.environ.get("DISCORD_WEBHOOK_URL"),
            line_channel_access_token=os.environ.get("LINE_CHANNEL_ACCESS_TOKEN"),
            line_user_id=os.environ.get("LINE_USER_ID"))
        notifier.notify(actionable, breadth, dry_run=dry_run, pages_url=self.pages_url)
        logger.info("=== 処理完了：選定%d、通知候補%d ===", len(data), len(actionable))
        return actionable


# ============================================================================
# 補助関数: CSV / JPX銘柄一覧 / 200銘柄の分割調査
# ============================================================================
def normalize_ticker(value):
    ticker = str(value).strip().upper()
    if re.fullmatch(r"(?:[0-9]{4}|[0-9]{3}[A-Z])", ticker):
        return ticker + ".T"
    return ticker if re.fullmatch(r"(?:[0-9]{4}|[0-9]{3}[A-Z])\.T", ticker) else None


def load_tickers_from_csv(path):
    df = pd.read_csv(path, header=None, dtype=str, encoding="utf-8-sig")
    tickers = [normalize_ticker(value) for value in df.iloc[:, 0].dropna()]
    return list(dict.fromkeys(t for t in tickers if t))


def load_selected_tickers(path):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if "ticker" not in df.columns:
        tickers = load_tickers_from_csv(path)
        return tickers, {}
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])
    names = dict(zip(df["ticker"], df["name"].fillna(df["ticker"]))) if "name" in df.columns else {}
    return df["ticker"].tolist(), names


def fetch_jpx_source(cache_dir="cache"):
    """月次のJPX公式一覧。ページのURLが変更された場合は --jpx-file で補う。"""
    dest = Path(cache_dir) / "jpx"
    dest.mkdir(parents=True, exist_ok=True)
    cached = list(dest.glob("listed_*.xls*"))
    recent = [f for f in cached if
              (datetime.now().timestamp() - f.stat().st_mtime) < 35 * 86400]
    if recent:
        return max(recent, key=lambda f: f.stat().st_mtime)
    r = requests.get(JPX_LIST_PAGE, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    links = re.findall(r'href\s*=\s*["\']([^"\']+\.xls[x]?(?:\?[^"\']*)?)["\']',
                       r.text, flags=re.I)
    links = [urljoin(JPX_LIST_PAGE, link) for link in links]
    links = [link for link in links if link.startswith("https://www.jpx.co.jp/")]
    links.sort(key=lambda x: ("data_j" not in x, x))
    if not links:
        raise RuntimeError("JPXのExcel URLを取得できません。公式ページから保存し --jpx-file で指定してください")
    link = links[0]
    binary = requests.get(link, timeout=50, headers={"User-Agent": "Mozilla/5.0"})
    binary.raise_for_status()
    extension = ".xlsx" if link.lower().split("?")[0].endswith("xlsx") else ".xls"
    path = dest / f"listed_{today_jst():%Y%m%d}{extension}"
    path.write_bytes(binary.content)
    return path


def load_jpx_list(path):
    path = Path(path)
    if path.suffix.lower() in (".xls", ".xlsx"):
        frame = pd.read_excel(path, dtype=str)
    else:
        frame = None
        for encoding in ("utf-8-sig", "cp932"):
            try:
                frame = pd.read_csv(path, dtype=str, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if frame is None:
            raise ValueError("JPX CSVを読み取れませんでした")
    for attempt in range(5):
        if "コード" in frame.columns:
            break
        frame.columns = frame.iloc[0].fillna("").astype(str)
        frame = frame.iloc[1:].reset_index(drop=True)
    if "コード" not in frame.columns:
        raise ValueError("JPX一覧に「コード」列がありません。別のExcelシートでないか確認")
    market = next((c for c in frame.columns if "市場・商品区分" in str(c)), None)
    name = next((c for c in frame.columns if "銘柄名" in str(c)), None)
    if market is None:
        raise ValueError("JPX一覧に「市場・商品区分」列がありません")
    mask = frame[market].fillna("").str.contains("内国株式", regex=False)
    mask &= frame[market].fillna("").str.contains("プライム|スタンダード|グロース", regex=True)
    candidates = frame.loc[mask, ["コード", market] + ([name] if name else [])].copy()
    candidates["ticker"] = candidates["コード"].map(normalize_ticker)
    candidates = candidates.dropna(subset=["ticker"]).drop_duplicates("ticker")
    candidates["name"] = (candidates[name].fillna(candidates["ticker"])
                          if name else candidates["ticker"])
    candidates["market"] = candidates[market]
    if len(candidates) < 1000:
        raise ValueError(f"内国普通株式が{len(candidates)}件しかないため、JPX一覧の形式を要確認")
    return candidates[["ticker", "name", "market"]].reset_index(drop=True)


def compute_universe_metrics(ticker, df, meta, capital):
    """勝率ではなく流動性・値動き・シグナル発生頻度で一次選定する。"""
    if len(df) < 120 or df.index[-1].date() != previous_session():
        return None
    indicators = TechnicalEngine.compute_indicators(df)
    recent = indicators.tail(240)
    latest = recent.iloc[-1]
    turnover = float((df["Close"] * df["Volume"]).tail(20).mean())
    atr_pct = float(latest["atr14"] / latest["Close"] * 100)
    close = float(latest["Close"])
    if not (turnover >= 300_000_000 and 2 <= atr_pct <= 8 and
            close * 100 <= capital and float(latest["atr14"]) * 1.5 * 100 <= capital * 0.01):
        return None
    a = (recent["dev5"] <= CONFIG.a_dev5_threshold) & (
        recent["close_open_ratio"] <= CONFIG.a_bear_candle_ratio)
    b = recent["dev3"].between(CONFIG.b_dev3_lower, CONFIG.b_dev3_upper)
    c = recent["pct_change"] >= CONFIG.c_pct_change_threshold
    d = ((recent["lower_shadow_ratio"] > CONFIG.d_lower_shadow_ratio) &
         (recent["pct_change"] <= CONFIG.d_pct_change_threshold))
    return dict(ticker=ticker, name=meta["name"], market=meta["market"],
                avg_turnover20=round(turnover), atr_pct=round(atr_pct, 3),
                buy_signal_days=int((a | b | d).sum()), short_signal_days=int(c.sum()),
                data_points=len(df), last_scanned=today_jst().isoformat(),
                last_bar=df.index[-1].date().isoformat())


def build_universe(jpx_file=None, cache_dir="cache", state_file="universe/universe_metrics.csv",
                   output_file="tickers_200.csv", batch_size=80, capital=1_000_000,
                   shortable_file=None):
    path = jpx_file or fetch_jpx_source(cache_dir)
    all_issues = load_jpx_list(path)
    state_path = Path(state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = pd.read_csv(state_path, dtype={"ticker": str})
    else:
        state = pd.DataFrame()
    known = set(state["ticker"]) if "ticker" in state else set()
    options = all_issues.to_dict("records")
    # 固定シャッフル: 数日分の分割処理でも業種・市場の先頭順に偏らない。
    random.Random(20260930).shuffle(options)
    pending = [x for x in options if x["ticker"] not in known]
    if pending:
        portion = pending[:batch_size]
    else:
        # 全銘柄調査後は前回調査日が古い順に、部分的に更新する。
        meta = {x["ticker"]: x for x in options}
        ordered = state.sort_values("last_scanned") if not state.empty else state
        portion = [meta[t] for t in ordered["ticker"] if t in meta][:batch_size]
    logger.info("JPX対象 %d件 / 過去調査 %d件 / 今回 %d件",
                len(options), len(known), len(portion))
    tickers = [x["ticker"] for x in portion]
    loader = DataLoader(tickers, cache_dir=cache_dir)
    prices = loader.fetch_all()
    records = []
    for meta in portion:
        t = meta["ticker"]
        measure = compute_universe_metrics(t, prices[t], meta, capital) if t in prices else None
        # 条件に合わない銘柄も走査済み記録として保持し、無限再取得しない。
        records.append(measure or {"ticker": t, "name": meta["name"], "market": meta["market"],
                                  "avg_turnover20": 0, "atr_pct": 0,
                                  "buy_signal_days": 0, "short_signal_days": 0,
                                  "data_points": len(prices.get(t, [])),
                                  "last_scanned": today_jst().isoformat(),
                                  "last_bar": (prices[t].index[-1].date().isoformat() if t in prices else "")})
    fresh = pd.DataFrame(records)
    if not state.empty:
        state = state[~state["ticker"].isin(tickers)]
    state = pd.concat([state, fresh], ignore_index=True).drop_duplicates("ticker", keep="last")
    state.to_csv(state_path, index=False)
    eligible = state[(state["avg_turnover20"] >= 300_000_000) &
                     (state["atr_pct"].between(2, 8))].copy()
    eligible["buy_signal_days"] = pd.to_numeric(eligible["buy_signal_days"], errors="coerce").fillna(0)
    eligible["short_signal_days"] = pd.to_numeric(eligible["short_signal_days"], errors="coerce").fillna(0)
    eligible["avg_turnover20"] = pd.to_numeric(eligible["avg_turnover20"], errors="coerce").fillna(0)
    buys = eligible.sort_values(["buy_signal_days", "avg_turnover20"], ascending=False)
    shortables = (set(load_tickers_from_csv(shortable_file))
                  if shortable_file and Path(shortable_file).exists() else set())
    shorts = (eligible[eligible["ticker"].isin(shortables)]
              .sort_values(["short_signal_days", "avg_turnover20"], ascending=False))
    buy_quota = 150 if len(shorts) >= 50 else 200
    selected = pd.concat([buys.head(buy_quota), shorts.head(50)]).drop_duplicates("ticker")
    if len(selected) < 200:
        extra = buys[~buys["ticker"].isin(selected["ticker"])]
        selected = pd.concat([selected, extra.head(200 - len(selected))])
    selected = selected.head(200)
    logger.info("条件適合 %d銘柄 / 暫定選定 %d銘柄 / 全市場進捗 %d/%d",
                len(eligible), len(selected), len(state), len(options))
    if len(selected) >= 200:
        selected.to_csv(output_file, index=False)
        logger.info("%d銘柄のリストを更新: %s（過去成績ではなく選定条件による）",
                    len(selected), output_file)
    else:
        logger.warning("200銘柄未満のため既存 %s は上書きしません。分割調査を継続", output_file)
    return selected


# ============================================================================
# エントリーポイント
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="デイトレ候補 v2 / 200銘柄自動選定")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--risk-pct", type=float, default=0.01)
    parser.add_argument("--portfolio-risk-pct", type=float, default=0.03)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--tickers-file", type=str, default=None)
    parser.add_argument("--breadth-file", type=str, default=None)
    parser.add_argument("--shortable-file", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-path", type=str, default="docs/index.html")
    parser.add_argument("--pages-url", type=str, default=os.environ.get("PAGES_URL"))
    parser.add_argument("--log-path", type=str, default="logs/signal_log.csv")
    parser.add_argument("--build-universe", action="store_true", help="JPXから200銘柄を分割自動選定")
    parser.add_argument("--jpx-file", type=str, default=None, help="公式JPX一覧のxls/xlsx/csv")
    parser.add_argument("--universe-batch-size", type=int, default=80)
    parser.add_argument("--universe-state", type=str, default="universe/universe_metrics.csv")
    parser.add_argument("--universe-output", type=str, default="tickers_200.csv")
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--max-gap-pct", type=float, default=5.0)
    parser.add_argument("--slippage-pct", type=float, default=0.001)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.capital <= 0 or not (0 < args.risk_pct <= 0.1):
        raise ValueError("capital > 0、0 < risk-pct <= 0.1 としてください")
    if not (0 < args.portfolio_risk_pct <= 0.1):
        raise ValueError("portfolio-risk-pct は 0～0.1 の範囲内")
    if args.max_positions < 1 or args.universe_batch_size < 1:
        raise ValueError("max-positionsとuniverse-batch-sizeは1以上")
    if not (0 < args.min_coverage <= 1):
        raise ValueError("min-coverage は0～1")
    if args.build_universe:
        build_universe(jpx_file=args.jpx_file, cache_dir=args.cache_dir,
                       state_file=args.universe_state, output_file=args.universe_output,
                       batch_size=args.universe_batch_size, capital=args.capital,
                       shortable_file=args.shortable_file)
        return
    chosen_file = args.tickers_file or (args.universe_output if Path(args.universe_output).exists() else None)
    if chosen_file:
        tickers, names = load_selected_tickers(chosen_file)
        logger.info("CSVから %d 銘柄: %s", len(tickers), chosen_file)
    else:
        tickers, names = list(dict.fromkeys(DEFAULT_TICKERS)), {}
        logger.warning("200銘柄CSV未作成。内蔵のサンプル %d 銘柄を使います", len(tickers))
    if not tickers:
        raise ValueError("対象銘柄が0件です。CSVを確認してください")
    system = TradingSystem(
        tickers=tickers, names=names, capital=args.capital, risk_pct=args.risk_pct,
        cache_dir=args.cache_dir, report_path=args.report_path or None,
        pages_url=args.pages_url or None, log_path=args.log_path or None,
        breadth_file=args.breadth_file, shortable_file=args.shortable_file,
        max_positions=args.max_positions, portfolio_risk_pct=args.portfolio_risk_pct,
        min_coverage=args.min_coverage, max_gap_pct=args.max_gap_pct,
        slippage_pct=args.slippage_pct)
    system.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
