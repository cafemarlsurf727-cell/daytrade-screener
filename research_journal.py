"""資金配分とは独立して全シグナルを検証する研究用ログ。

売買注文・実約定ではなく、翌営業日の確定日足を用いた仮想結果。
見送り条件（寄り付きギャップ等）は元の売買モデルに合わせて別集計する。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

JST = timezone(timedelta(hours=9))
RESEARCH_VERSION = "research-v1-open-slippage"


class ResearchSignalJournal:
    COLUMNS = [
        "signal_date", "as_of_date", "ticker", "name", "strategy_code",
        "matched_strategies", "strategy_label", "side", "signal_close",
        "stop_loss", "allocation_status", "allocation_reason", "model_version",
        "status", "actual_open", "actual_high", "actual_low", "actual_close",
        "gap_pct", "modeled_entry", "modeled_exit", "stopped_out",
        "pnl_per_share", "return_pct", "r_multiple", "resolved_at", "skip_reason",
    ]

    def __init__(self, log_path, max_gap_pct=5.0, slippage_pct=0.001):
        self.path = Path(log_path)
        self.max_gap_pct = max_gap_pct
        self.slippage_pct = slippage_pct

    def load(self):
        if not self.path.exists():
            return pd.DataFrame(columns=self.COLUMNS)
        try:
            df = pd.read_csv(self.path, dtype={"ticker": str})
            missing = set(self.COLUMNS) - set(df.columns)
            if missing:
                raise ValueError(f"必須列が欠落: {sorted(missing)}")
            return df[self.COLUMNS]
        except Exception as exc:
            raise RuntimeError(
                f"研究ログの読み取りに失敗。データ保護のため処理を中止します: {exc}"
            ) from exc

    def save(self, df):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(self.path)

    def append(self, df, candidates, signal_date, as_of_date):
        """技術的に条件一致した全件を登録。配分不可でも0株に置換しない。"""
        existing = set(zip(
            df["signal_date"].astype(str),
            df["ticker"].astype(str),
            df["strategy_code"].astype(str),
            df["model_version"].astype(str),
        ))
        new_rows = []
        for item in candidates:
            key = (str(signal_date), str(item["ticker"]),
                   str(item["strategy_code"]), RESEARCH_VERSION)
            if key in existing:
                continue
            row = dict.fromkeys(self.COLUMNS, pd.NA)
            row.update({
                "signal_date": str(signal_date),
                "as_of_date": str(as_of_date),
                "ticker": item["ticker"],
                "name": item["name"],
                "strategy_code": item["strategy_code"],
                "matched_strategies": item.get("matched_strategies", item["strategy_code"]),
                "strategy_label": item["strategy_label"],
                "side": item["side"],
                "signal_close": float(item["entry_price"]),
                "stop_loss": float(item["stop_loss"]),
                "allocation_status": item.get("allocation_status", "research_only"),
                "allocation_reason": item.get("allocation_reason"),
                "model_version": RESEARCH_VERSION,
                "status": "pending",
            })
            new_rows.append(row)
            existing.add(key)
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows, columns=self.COLUMNS)],
                           ignore_index=True)
        return df, len(new_rows)

    def resolve(self, df, prices, *, today=None):
        """翌営業日以降にsignal_dateの確定OHLCで仮想売買結果を算定する。"""
        if df.empty:
            return df, 0
        today = today or datetime.now(JST).date()
        df = df.copy()
        # CSVの型推定に依存せず、更新列を単一値で安全に書き換える。
        for col in self.COLUMNS[self.COLUMNS.index("status"):]:
            df[col] = df[col].astype(object)
        resolved_count = 0
        for idx in df[df["status"] == "pending"].index:
            row = df.loc[idx]
            signal_day = str(row["signal_date"])[:10]
            ticker = str(row["ticker"])
            if signal_day >= today.isoformat() or ticker not in prices:
                continue
            bars = prices[ticker]
            match = [str(day)[:10] == signal_day for day in bars.index]
            if not any(match):
                continue
            candle = bars.loc[match].iloc[0]
            op, hi, lo, cl = (float(candle[c]) for c in
                              ("Open", "High", "Low", "Close"))
            prev_close = float(row["signal_close"])
            stop = float(row["stop_loss"])
            side = str(row["side"])
            for col, val in (("actual_open", op), ("actual_high", hi),
                             ("actual_low", lo), ("actual_close", cl)):
                df.at[idx, col] = val

            reason = None
            gap = None
            if not all(np.isfinite(n) and n > 0 for n in
                       (op, hi, lo, cl, prev_close, stop)):
                reason = "価格データ異常"
            else:
                gap = (op / prev_close - 1) * 100
                if abs(gap) > self.max_gap_pct:
                    reason = f"寄り付きギャップ{gap:+.2f}%が上限超過"
                elif (side == "buy" and op <= stop) or (
                    side == "sell" and op >= stop
                ):
                    reason = "寄り付き時点で事前損切り価格を超過"
                elif side not in ("buy", "sell"):
                    reason = "売買方向が不明"
            df.at[idx, "gap_pct"] = gap
            df.at[idx, "resolved_at"] = datetime.now(JST).isoformat(timespec="seconds")
            if reason:
                df.at[idx, "status"] = "skipped"
                df.at[idx, "skip_reason"] = reason
                resolved_count += 1
                continue

            stopped = lo <= stop if side == "buy" else hi >= stop
            target_exit = stop if stopped else cl
            entry = op * (1 + self.slippage_pct if side == "buy"
                          else 1 - self.slippage_pct)
            exit_ = target_exit * (1 - self.slippage_pct if side == "buy"
                                   else 1 + self.slippage_pct)
            pnl = exit_ - entry if side == "buy" else entry - exit_
            risk = abs(op - stop)
            df.at[idx, "modeled_entry"] = entry
            df.at[idx, "modeled_exit"] = exit_
            df.at[idx, "stopped_out"] = bool(stopped)
            df.at[idx, "pnl_per_share"] = pnl
            df.at[idx, "return_pct"] = 100 * pnl / op
            df.at[idx, "r_multiple"] = pnl / risk
            df.at[idx, "status"] = "resolved"
            resolved_count += 1
        return df, resolved_count

    @staticmethod
    def summarize(df):
        stats = {
            "counts": {"total": 0, "pending": 0, "resolved": 0, "skipped": 0},
            "overall": None, "by_strategy": {},
        }
        if df.empty:
            return stats
        current = df[df["model_version"] == RESEARCH_VERSION]
        stats["counts"] = {
            "total": len(current),
            "pending": int((current["status"] == "pending").sum()),
            "resolved": int((current["status"] == "resolved").sum()),
            "skipped": int((current["status"] == "skipped").sum()),
        }
        resolved = current[current["status"] == "resolved"].copy()
        for col in ("pnl_per_share", "return_pct", "r_multiple"):
            resolved[col] = pd.to_numeric(resolved[col], errors="coerce")
        resolved = resolved.dropna(subset=["pnl_per_share", "return_pct", "r_multiple"])
        if resolved.empty:
            return stats

        def agg(rows):
            return {
                "n": len(rows),
                "win_rate": float((rows["pnl_per_share"] > 0).mean()) * 100,
                "avg_pnl_per_share": float(rows["pnl_per_share"].mean()),
                "avg_return_pct": float(rows["return_pct"].mean()),
                "avg_r": float(rows["r_multiple"].mean()),
            }

        stats["overall"] = agg(resolved)
        stats["by_strategy"] = {
            code: agg(part) for code, part in resolved.groupby("strategy_code")
        }
        return stats


def render_research_stats_html(stats):
    """既存の「資金配分候補の仮想検証」パネルと分離して描画する。"""
    if stats is None:
        return ""
    counts = stats["counts"]

    def metrics(item):
        return (
            f"勝率 {item['win_rate']:.1f}% / "
            f"平均損益 {item['avg_pnl_per_share']:+.2f}円/株 / "
            f"平均収益率 {item['avg_return_pct']:+.2f}% / "
            f"期待値 {item['avg_r']:+.2f}R（n={item['n']}）"
        )

    overall = stats["overall"]
    headline = (
        f'<p class="breadth-value">{metrics(overall)}</p>' if overall else
        '<p class="empty-state">まだ検証済みシグナルはありません。</p>'
    )
    name_map = {
        "A": "逆張り買い", "B": "急落後の逆張り買い",
        "C": "急騰株空売り", "D": "下髭サポート買い",
    }
    rows = "".join(
        '<div class="stat-row">'
        f'<span class="stat-label">{code}: {name_map[code]}</span>'
        f'<span class="stat-value">{metrics(stats["by_strategy"][code]) if code in stats["by_strategy"] else "検証済み0件"}</span>'
        '</div>' for code in ("A", "B", "C", "D")
    )
    return (
        '<section class="breadth-panel">'
        '<p class="breadth-label">全候補の研究用・仮想検証（資金額に依存しない）</p>'
        f'<p class="breadth-label">記録 {counts["total"]}件 / '
        f'検証済み {counts["resolved"]}件 / 判定待ち {counts["pending"]}件 / '
        f'寄り付き条件等で見送り {counts["skipped"]}件</p>'
        f'{headline}<div class="stat-rows">{rows}</div>'
        '<p class="breadth-label" style="margin-top:8px;">'
        '前日終値で発見、翌取引日寄り付きで仮想エントリーし同日手仕舞い。'
        'ATR損切り、売買各側0.1%の仮想スリッページを含む。'
        '寄り付きギャップ5%超等の見送りは記録するが勝率・期待値には含めない。'
        '平均損益は1株当たり、期待値は平均R（1トレードの損益÷寄り付き時の損切り幅）。'
        '空売り規制・実際の約定・手数料・税金は反映しない。'
        '</p></section>'
    )
