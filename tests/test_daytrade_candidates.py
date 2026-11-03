"""Regression tests for candidate discovery, allocation visibility, and interim universe."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import daytrade_screener as ds


def sample_prices(close_last=80.0, open_last=90.0):
    days = pd.bdate_range(end="2026-10-30", periods=30)
    frame = pd.DataFrame({
        "Open": [100.0] * len(days),
        "High": [102.0] * len(days),
        "Low": [99.0] * len(days),
        "Close": [100.0] * len(days),
        "Volume": [1_000_000] * len(days),
    }, index=days)
    frame.iloc[-1, frame.columns.get_loc("Open")] = open_last
    frame.iloc[-1, frame.columns.get_loc("High")] = max(open_last, close_last) + 5
    frame.iloc[-1, frame.columns.get_loc("Low")] = min(open_last, close_last) - 3
    frame.iloc[-1, frame.columns.get_loc("Close")] = close_last
    return frame


class ProvisionalUniverseTests(unittest.TestCase):
    def test_incomplete_universe_is_used_without_relaxing_eligibility(self):
        rows = pd.DataFrame({
            "ticker": ["1111.T", "2222.T", "3333.T", "4444.T"],
            "name": ["Eligible A", "Low turnover", "Eligible B", "Bad ATR"],
            "avg_turnover20": [400_000_000, 100_000_000, 800_000_000, 500_000_000],
            "atr_pct": [3.0, 3.0, 5.0, 10.0],
            "buy_signal_days": [3, 8, 1, 12],
            "short_signal_days": [0, 0, 0, 0],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "universe_metrics.csv"
            rows.to_csv(path, index=False)
            tickers, names = ds.load_provisional_tickers(path)
        self.assertEqual(tickers, ["1111.T", "3333.T"])
        self.assertEqual(names["1111.T"], "Eligible A")

    def test_missing_universe_uses_empty_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tickers, names = ds.load_provisional_tickers(Path(tmp) / "missing.csv")
        self.assertEqual(tickers, [])
        self.assertEqual(names, {})


class CandidateVisibilityTests(unittest.TestCase):
    def run_screen(self, capital, close_last=80.0, open_last=90.0):
        with tempfile.TemporaryDirectory() as tmp:
            outfile = Path(tmp) / "report.html"
            trading = ds.TradingSystem(
                tickers=["1111.T"], capital=capital, risk_pct=0.01,
                cache_dir=str(Path(tmp) / "cache"),
                report_path=str(outfile), log_path=None)
            with patch.object(ds, "is_trading_day", return_value=True), \
                 patch.object(ds.DataLoader, "fetch_all",
                              return_value={"1111.T": sample_prices(close_last, open_last)}), \
                 patch.object(ds.Notifier, "notify") as notify:
                allocated = trading.run(dry_run=True)
            html = outfile.read_text(encoding="utf-8")
            displayed = notify.call_args.args[0]
        return allocated, displayed, html

    def test_raw_signal_survives_risk_rejection_for_report_and_notice(self):
        allocated, displayed, html = self.run_screen(capital=10_000)
        self.assertEqual(allocated, [])
        self.assertEqual(len(displayed), 1)
        self.assertEqual(displayed[0]["allocation_status"], "research_only")
        self.assertEqual(displayed[0]["max_shares"], 0)
        self.assertIn("許容損失", displayed[0]["warning"])
        self.assertIn("条件一致 1銘柄", html)
        self.assertIn("参考候補（資金配分なし）", html)
        self.assertIn("1111.T", html)

    def test_funded_signal_remains_allocatable(self):
        allocated, displayed, html = self.run_screen(capital=1_000_000)
        self.assertEqual(len(allocated), 1)
        self.assertEqual(len(displayed), 1)
        self.assertEqual(displayed[0]["allocation_status"], "allocated")
        self.assertGreater(displayed[0]["max_shares"], 0)
        self.assertIn("資金配分候補 1銘柄", html)

    def test_short_signal_without_shortable_list_stays_research_only(self):
        allocated, displayed, html = self.run_screen(
            capital=1_000_000, close_last=125.0, open_last=118.0)
        self.assertEqual(allocated, [])
        self.assertEqual(len(displayed), 1)
        self.assertEqual(displayed[0]["strategy_code"], "C")
        self.assertEqual(displayed[0]["allocation_status"], "research_only")
        self.assertIn("空売り在庫・規制", displayed[0]["warning"])
        self.assertIn("参考候補（資金配分なし）", html)


if __name__ == "__main__":
    unittest.main()
