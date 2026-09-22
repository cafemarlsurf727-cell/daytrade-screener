#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 デイトレード自動スクリーニング＆通知システム (daytrade_screener.py)
============================================================================

■ 概要
    前日までの株価データを取得し、テクニカル指標に基づく複数の売買シグナル
    （逆張り買い／押し目買い／急騰空売り／下髭サポート買い）を自動判定して、
    Discord または LINE に「当日朝の候補銘柄一覧」を通知するシステムです。

    このシステムが検出するのはあくまで「エントリー候補」です。
    最終的な発注判断・執行は必ず人間（あなた自身）が行ってください。

    ※本コードは教育・研究目的のサンプルです。実運用前に必ず十分な
      バックテストとペーパートレードを行い、自己責任でご利用ください。
      利益を保証するものではありません。

----------------------------------------------------------------------------
■ 環境構築手順
----------------------------------------------------------------------------
    1) Python 3.10 以上を用意する

    2) 必要ライブラリをインストール
        $ pip install pandas numpy yfinance requests

    3) 通知先の環境変数を設定する（どちらか、または両方）
        - Discord を使う場合:
            export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/xxxx/yyyy"

        - LINE を使う場合（重要な注意点あり。下記参照）:
            export LINE_CHANNEL_ACCESS_TOKEN="xxxxxxxx"
            export LINE_USER_ID="Uxxxxxxxxxxxxxxxx"

        【重要】旧サービス「LINE Notify」は 2025年3月31日 に終了しています。
        本コードは後継である「LINE Messaging API」の push メッセージ機能を
        使用しています（LINE Developers コンソールで Messaging API
        チャネルを作成し、チャネルアクセストークンと自分の userId を
        取得してください）。

----------------------------------------------------------------------------
■ 実行方法
----------------------------------------------------------------------------
    基本実行（総資金100万円、1トレード許容リスク1%）:
        $ python daytrade_screener.py --capital 1000000 --risk-pct 0.01

    銘柄リストを自前のCSV（1列目にYahoo!Financeティッカー、例: 7203.T）
    から読み込む場合:
        $ python daytrade_screener.py --tickers-file my_tickers.csv

    通知を送らずコンソール確認だけしたい場合（テスト用）:
        $ python daytrade_screener.py --dry-run

----------------------------------------------------------------------------
■ Cron設定例（毎営業日 朝7:30 に実行し、寄り付き前に通知を受け取る）
----------------------------------------------------------------------------
    30 7 * * 1-5 cd /path/to/project && \
        /usr/bin/python3 daytrade_screener.py --capital 1000000 >> log.txt 2>&1

----------------------------------------------------------------------------
■ GitHub Actions 設定例（.github/workflows/daytrade_screener.yml）
----------------------------------------------------------------------------
    name: daytrade-screener
    on:
      schedule:
        # UTC 22:30 = 日本時間 翌7:30（平日のみ。UTC基準なので日付ずれに注意）
        - cron: "30 22 * * 0-4"
      workflow_dispatch: {}
    permissions:
      contents: write   # レポートHTMLをリポジトリに書き戻すために必要
    jobs:
      run:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: "3.11"
          - run: pip install pandas numpy yfinance requests
          - run: python daytrade_screener.py --capital 1000000
            env:
              DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
              LINE_CHANNEL_ACCESS_TOKEN: ${{ secrets.LINE_CHANNEL_ACCESS_TOKEN }}
              LINE_USER_ID: ${{ secrets.LINE_USER_ID }}
              # 例: https://ユーザー名.github.io/リポジトリ名/
              PAGES_URL: "https://<ユーザー名>.github.io/<リポジトリ名>/"
          - name: レポートHTMLをコミット
            run: |
              git config user.name "github-actions[bot]"
              git config user.email "github-actions[bot]@users.noreply.github.com"
              git add docs/index.html
              git diff --staged --quiet || git commit -m "chore: update daytrade report"
              git push

----------------------------------------------------------------------------
■ GitHub Pages の有効化（初回のみ・GUI操作）
----------------------------------------------------------------------------
    1) リポジトリの Settings → Pages を開く
    2) "Build and deployment" の Source を "Deploy from a branch" にする
    3) Branch を "main" / フォルダを "/docs" にして Save
    4) 数分後、https://<ユーザー名>.github.io/<リポジトリ名>/ でレポートが閲覧可能になる
       （上記ワークフローの PAGES_URL をこのURLに合わせておくこと）

============================================================================
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import time
from datetime import datetime
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
    "8031.T", "8001.T", "8306.T", "4568.T", "6902.T", "6503.T", "7741.T",
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
    atr_multiplier: float = 1.5
    unit_shares: int = 100  # 日本株の単元株数（多くは100株単位）


CONFIG = StrategyConfig()


# ============================================================================
# 1. DataLoader モジュール
# ============================================================================
class DataLoader:
    """
    株価データの一括取得・キャッシュを担当するクラス。

    - yfinance を使って日足データ（デフォルト60日分）を取得する。
    - 当日中に同じ銘柄を再取得しないよう、ローカルにCSVキャッシュを持つ。
    - 個別銘柄の取得失敗（上場廃止、通信エラー等）が全体を止めないよう、
      1銘柄ずつ例外処理を行い、失敗した銘柄はスキップしてログに残す。
    """

    def __init__(
        self,
        tickers: list[str],
        period_days: int = 60,
        cache_dir: str = "cache",
        max_retries: int = 2,
        retry_wait_sec: float = 1.5,
    ) -> None:
        self.tickers = tickers
        self.period_days = period_days
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_wait_sec = retry_wait_sec

    def _cache_path(self, ticker: str) -> Path:
        today_str = datetime.now().strftime("%Y%m%d")
        return self.cache_dir / f"{ticker}_{today_str}.csv"

    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(ticker)
        if path.exists():
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                if not df.empty:
                    return df
            except Exception as exc:  # noqa: BLE001
                logger.warning("キャッシュ読み込み失敗 (%s): %s", ticker, exc)
        return None

    def _save_to_cache(self, ticker: str, df: pd.DataFrame) -> None:
        try:
            df.to_csv(self._cache_path(ticker))
        except Exception as exc:  # noqa: BLE001
            logger.warning("キャッシュ保存失敗 (%s): %s", ticker, exc)

    def _fetch_one(self, ticker: str) -> Optional[pd.DataFrame]:
        """1銘柄分の日足データを取得する（リトライ付き）。"""
        cached = self._load_from_cache(ticker)
        if cached is not None:
            return cached

        for attempt in range(1, self.max_retries + 1):
            try:
                df = yf.download(
                    ticker,
                    period=f"{self.period_days}d",
                    interval="1d",
                    progress=False,
                    auto_adjust=False,
                )
                # yfinanceがMultiIndex列を返す場合があるため正規化
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                if df is None or df.empty or len(df) < 15:
                    logger.warning(
                        "データ不足のためスキップ: %s (取得件数=%s)",
                        ticker,
                        0 if df is None else len(df),
                    )
                    return None

                df = df.dropna(subset=["Open", "High", "Low", "Close"])
                self._save_to_cache(ticker, df)
                return df

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "取得エラー (%s) 試行%d/%d: %s",
                    ticker, attempt, self.max_retries, exc,
                )
                time.sleep(self.retry_wait_sec)

        logger.error("最終的に取得失敗: %s", ticker)
        return None

    def fetch_all(self) -> dict[str, pd.DataFrame]:
        """全銘柄のデータを取得し、{ティッカー: DataFrame} の辞書で返す。"""
        result: dict[str, pd.DataFrame] = {}
        total = len(self.tickers)
        for i, ticker in enumerate(self.tickers, start=1):
            logger.info("取得中 (%d/%d): %s", i, total, ticker)
            df = self._fetch_one(ticker)
            if df is not None:
                result[ticker] = df
        logger.info("データ取得完了: %d/%d 銘柄成功", len(result), total)
        return result


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
    def compute_market_breadth(data_dict: dict[str, pd.DataFrame]) -> dict[str, float]:
        """
        市場全体の地合いを判定する指標をまとめて計算する。

        - ad_ratio       : 騰落レシオ(%) = (直近25日の値上がり銘柄数合計 /
                            値下がり銘柄数合計) × 100 をスクリーニング対象
                            ユニバーニバースで簡易的に算出したもの。
                            ※本来は市場全銘柄で計算する指標だが、ここでは
                              取得済みユニバースでの近似値として扱う。
        - avg_dev5        : ユニバース全体の5日移動平均乖離率の平均値(%)
        """
        advances = 0
        declines = 0
        dev5_list: list[float] = []

        for ticker, df in data_dict.items():
            if len(df) < 11:
                continue
            indicators = TechnicalEngine.compute_indicators(df)
            recent = indicators.tail(25)
            advances += int((recent["pct_change"] > 0).sum())
            declines += int((recent["pct_change"] < 0).sum())

            last_dev5 = indicators["dev5"].iloc[-1]
            if pd.notna(last_dev5):
                dev5_list.append(float(last_dev5))

        ad_ratio = (advances / declines * 100.0) if declines > 0 else 100.0
        avg_dev5 = float(np.mean(dev5_list)) if dev5_list else 0.0

        logger.info(
            "市場地合い: 騰落レシオ=%.1f%%, 平均5日乖離率=%.2f%%",
            ad_ratio, avg_dev5,
        )
        return {"ad_ratio": ad_ratio, "avg_dev5": avg_dev5}


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

    def screen(
        self,
        data_dict: dict[str, pd.DataFrame],
        names: Optional[dict[str, str]] = None,
        suppress_reverse_buy: bool = False,
    ) -> list[dict]:
        """
        全銘柄をスキャンし、該当したシグナルのリストを返す。

        suppress_reverse_buy が True の場合、Strategy A / B / D
        （逆張り・押し目買い系）は地合い過熱のため抑制し、
        Strategy C（空売り）のみ有効にする（NoviceGuardrail 用）。
        """
        signals: list[dict] = []
        names = names or {}

        for ticker, df in data_dict.items():
            indicators = TechnicalEngine.compute_indicators(df)
            row = self._latest_row(indicators)
            if row is None:
                continue

            entry_price = float(row["Close"])
            atr = float(row["atr14"])
            candidate_strategies = []

            if not suppress_reverse_buy and self._check_strategy_a(row):
                candidate_strategies.append(("A", "逆張り買い", "buy"))
            if not suppress_reverse_buy and self._check_strategy_b(row):
                candidate_strategies.append(("B", "押し目買い", "buy"))
            if self._check_strategy_c(row):
                candidate_strategies.append(("C", "急騰株空売り", "sell"))
            if not suppress_reverse_buy and self._check_strategy_d(row):
                candidate_strategies.append(("D", "下髭サポート買い", "buy"))

            for code, label, side in candidate_strategies:
                signals.append(
                    {
                        "ticker": ticker,
                        "name": names.get(ticker, ticker),
                        "strategy_code": code,
                        "strategy_label": label,
                        "side": side,
                        "entry_price": entry_price,
                        "atr": atr,
                        "dev3": float(row["dev3"]),
                        "dev5": float(row["dev5"]),
                        "pct_change": float(row["pct_change"]),
                        "lower_shadow_ratio": float(row["lower_shadow_ratio"]),
                    }
                )

        logger.info("シグナル検出: %d 件", len(signals))
        return signals


# ============================================================================
# 4. NoviceGuardrail & RiskEngine モジュール
# ============================================================================
class RiskEngine:
    """
    初心者保護のためのガードレールと、資金管理計算を担当するクラス。

    - 地合いフィルター: 極端な買われすぎ相場では逆張り買い系シグナルを抑制する
    - ATRベースの動的損切り: ATR×1.5倍を損切り目安とし、狭すぎる損切りに
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
        overheated = breadth["ad_ratio"] >= self.config.overheated_ad_ratio
        if overheated:
            logger.info(
                "地合いフィルター発動: 騰落レシオ %.1f%% >= %.1f%% のため、"
                "逆張り買い系シグナルを抑制します。",
                breadth["ad_ratio"], self.config.overheated_ad_ratio,
            )
        return overheated

    def calc_stop_loss(self, entry_price: float, atr: float, side: str) -> float:
        """ATR×1.5倍の位置に損切り価格を算出する。"""
        offset = atr * self.config.atr_multiplier
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
        stop_loss = self.calc_stop_loss(signal["entry_price"], signal["atr"], signal["side"])
        sizing = self.calc_position_size(signal["entry_price"], stop_loss)
        signal = dict(signal)
        signal["stop_loss"] = stop_loss
        signal.update(sizing)
        return signal


# ============================================================================
# 5. Notification モジュール
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
        today_str = datetime.now().strftime("%Y-%m-%d")
        lines = [
            f"【デイトレ候補銘柄】{today_str}",
            f"地合い: 騰落レシオ {breadth['ad_ratio']:.1f}% / "
            f"平均5日乖離率 {breadth['avg_dev5']:.2f}%",
            "※すべて当日大引け(15:00)で全決済すること。持ち越し厳禁。",
            "",
        ]

        if not signals:
            lines.append("本日は条件を満たす候補銘柄がありませんでした。無理に売買しないこと。")
            return "\n".join(lines)

        for s in signals:
            side_label = "買い" if s["side"] == "buy" else "空売り"
            lines.append(
                f"■ {s['name']} ({s['ticker']}) [{s['strategy_label']} / {side_label}]\n"
                f"  エントリー目安: {s['entry_price']:.1f}円\n"
                f"  損切り目安    : {s['stop_loss']:.1f}円 (ATR×1.5)\n"
                f"  推奨株数      : {s['max_shares']}株"
                f"（概算コスト {s['estimated_cost']:,.0f}円）\n"
                f"  想定リスク額  : {s['risk_amount']:,.0f}円\n"
            )
            if s.get("warning"):
                lines.append(f"  ⚠ {s['warning']}\n")

        lines.append("必ず15:00までに手仕舞いすること。損切りラインは厳守。")
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
        today_str = datetime.now().strftime("%Y-%m-%d")
        buy_count = sum(1 for s in signals if s["side"] == "buy")
        sell_count = sum(1 for s in signals if s["side"] == "sell")

        lines = [f"【デイトレ候補】{today_str}"]
        if signals:
            lines.append(f"検出 {len(signals)}件（買い{buy_count} / 空売り{sell_count}）")
        else:
            lines.append("本日は該当銘柄なし。無理に売買しないこと。")
        lines.append(f"地合い(騰落レシオ): {breadth['ad_ratio']:.1f}%")
        lines.append("※15:00までに全決済。")
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
# 6. ReportGenerator モジュール（GitHub Pages 用レポートHTML生成）
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
            <div><dt>エントリー目安</dt><dd>{s['entry_price']:.1f} 円</dd></div>
            <div><dt>損切り目安 (ATR×1.5)</dt><dd>{s['stop_loss']:.1f} 円</dd></div>
            <div><dt>推奨株数</dt><dd>{s['max_shares']:,} 株</dd></div>
            <div><dt>概算コスト</dt><dd>{s['estimated_cost']:,.0f} 円</dd></div>
            <div><dt>想定リスク額</dt><dd>{s['risk_amount']:,.0f} 円</dd></div>
            <div><dt>前日比</dt><dd>{s['pct_change']:+.2f} %</dd></div>
          </dl>
          {warning_html}
        </article>
        """

    def build_html(
        self,
        signals: list[dict],
        breadth: dict[str, float],
        capital: float,
        risk_pct: float,
    ) -> str:
        today_str = datetime.now().strftime("%Y-%m-%d (%a)")
        cards_html = "\n".join(self._render_card(s) for s in signals)
        if not signals:
            cards_html = (
                '<p class="empty-state">本日は条件を満たす候補銘柄がありません。'
                "無理に売買しないこと。</p>"
            )

        ad_ratio = breadth["ad_ratio"]
        # 騰落レシオを 0-200% のゲージにマッピング（100%を中央基準に）
        gauge_pct = max(0.0, min(100.0, (ad_ratio / 200.0) * 100.0))
        overheated = ad_ratio >= CONFIG.overheated_ad_ratio

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
      ⚠ 全ポジション、当日大引け(15:00)までに手仕舞いすること。持ち越し厳禁。
    </div>

    <section class="breadth-panel">
      <div class="breadth-row">
        <span class="breadth-label">市場地合い（騰落レシオ）</span>
        <span class="breadth-value">{ad_ratio:.1f}%</span>
      </div>
      <div class="gauge"><div class="gauge-fill"></div></div>
      <div class="breadth-row" style="margin-top:8px;">
        <span class="breadth-label">平均5日移動平均乖離率</span>
        <span class="breadth-value">{breadth['avg_dev5']:.2f}%</span>
      </div>
      {'<span class="overheated-tag">買われすぎ：逆張り買い系シグナルを抑制中</span>' if overheated else ''}
      <div class="breadth-row" style="margin-top:8px;">
        <span class="breadth-label">総資金 / 許容リスク</span>
        <span class="breadth-value">{capital:,.0f}円 / {risk_pct * 100:.1f}%</span>
      </div>
    </section>

    <section class="grid">
      {cards_html}
    </section>

    <footer>
      本レポートは教育・研究目的のサンプル出力です。投資判断は自己責任で行ってください。<br>
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
# 7. TradingSystem（全体オーケストレーション）
# ============================================================================
class TradingSystem:
    """
    DataLoader → TechnicalEngine → SignalScreener → RiskEngine → Notifier
    の一連の処理を実行する統括クラス。
    """

    def __init__(
        self,
        tickers: list[str],
        capital: float,
        risk_pct: float,
        cache_dir: str = "cache",
        names: Optional[dict[str, str]] = None,
        report_path: Optional[str] = None,
        pages_url: Optional[str] = None,
    ) -> None:
        self.tickers = tickers
        self.names = names or {}
        self.capital = capital
        self.risk_pct = risk_pct
        self.report_path = report_path
        self.pages_url = pages_url
        self.data_loader = DataLoader(tickers=tickers, cache_dir=cache_dir)
        self.screener = SignalScreener()
        self.risk_engine = RiskEngine(capital=capital, risk_pct=risk_pct)
        self.report_generator = ReportGenerator()

    def run(self, dry_run: bool = False) -> list[dict]:
        logger.info("=== デイトレード・スクリーニング開始 ===")

        # 1) データ取得
        data_dict = self.data_loader.fetch_all()
        if not data_dict:
            logger.error("有効な株価データを1件も取得できませんでした。処理を終了します。")
            return []

        # 2) 市場全体の地合い判定
        breadth = TechnicalEngine.compute_market_breadth(data_dict)
        suppress_reverse_buy = self.risk_engine.should_suppress_reverse_buy(breadth)

        # 3) シグナル判定
        raw_signals = self.screener.screen(
            data_dict, names=self.names, suppress_reverse_buy=suppress_reverse_buy
        )

        # 4) リスク管理情報の付与（損切り・ポジションサイズ）
        enriched_signals = [self.risk_engine.enrich_signal(s) for s in raw_signals]

        # 資金を確保できない（max_shares=0）シグナルは通知から除外しつつログに残す
        actionable = [s for s in enriched_signals if s["max_shares"] > 0]
        skipped = len(enriched_signals) - len(actionable)
        if skipped:
            logger.info("ポジションサイズ0のため通知から除外: %d 件", skipped)

        # 5) レポートHTML生成（GitHub Pages公開用）
        if self.report_path:
            html = self.report_generator.build_html(
                actionable, breadth, capital=self.capital, risk_pct=self.risk_pct
            )
            self.report_generator.save(html, self.report_path)

        # 6) 通知（pages_url指定時はLINEに詳細ページへのリンクを添える）
        notifier = Notifier(
            discord_webhook_url=os.environ.get("DISCORD_WEBHOOK_URL"),
            line_channel_access_token=os.environ.get("LINE_CHANNEL_ACCESS_TOKEN"),
            line_user_id=os.environ.get("LINE_USER_ID"),
        )
        notifier.notify(actionable, breadth, dry_run=dry_run, pages_url=self.pages_url)

        logger.info("=== デイトレード・スクリーニング終了 ===")
        return actionable


# ============================================================================
# 補助関数
# ============================================================================
def load_tickers_from_csv(path: str) -> list[str]:
    """
    CSVファイルから銘柄コードを読み込む。
    1列目にYahoo!Finance形式のティッカー（例: 7203.T）が入っている前提。
    ヘッダー行の有無は自動判定を試みる。
    """
    df = pd.read_csv(path, header=None, dtype=str)
    values = df.iloc[:, 0].dropna().tolist()
    tickers = [v.strip() for v in values if v.strip().upper() not in ("TICKER", "CODE", "銘柄コード")]
    return tickers


# ============================================================================
# エントリーポイント
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="デイトレード自動スクリーニング＆通知システム")
    parser.add_argument("--capital", type=float, default=1_000_000, help="総資金（円）。デフォルト100万円")
    parser.add_argument("--risk-pct", type=float, default=0.01, help="1トレードあたりの許容リスク比率。デフォルト0.01(1%)")
    parser.add_argument("--tickers-file", type=str, default=None, help="銘柄コードCSVファイルのパス")
    parser.add_argument("--cache-dir", type=str, default="cache", help="キャッシュ保存先ディレクトリ")
    parser.add_argument("--dry-run", action="store_true", help="通知を送らずコンソール表示のみ行う")
    parser.add_argument(
        "--report-path",
        type=str,
        default="docs/index.html",
        help="GitHub Pages公開用HTMLの出力先。空文字を指定するとレポート生成をスキップする。",
    )
    parser.add_argument(
        "--pages-url",
        type=str,
        default=os.environ.get("PAGES_URL"),
        help="公開済みGitHub PagesのURL。指定するとLINE通知が短い要約＋リンクになる。"
        "環境変数 PAGES_URL からも取得可能。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tickers_file:
        tickers = load_tickers_from_csv(args.tickers_file)
        logger.info("銘柄リストを読み込みました: %d 銘柄 (%s)", len(tickers), args.tickers_file)
    else:
        tickers = DEFAULT_TICKERS
        logger.warning(
            "銘柄リストが指定されていないため、同梱のサンプル銘柄(%d件)を使用します。"
            "本格運用時は --tickers-file で東証全銘柄リストを指定してください。",
            len(tickers),
        )

    system = TradingSystem(
        tickers=tickers,
        capital=args.capital,
        risk_pct=args.risk_pct,
        cache_dir=args.cache_dir,
        report_path=args.report_path or None,
        pages_url=args.pages_url or None,
    )
    system.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
