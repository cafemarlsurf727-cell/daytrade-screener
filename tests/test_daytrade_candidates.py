"""Regression tests for candidate discovery, allocation visibility, and interim universe."""
import tempfile
import unittest
from datetime import date
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


def universe_prices(end_day):
    days = pd.bdate_range(end=end_day, periods=140)
    return pd.DataFrame({
        "Open": [100.0] * len(days), "High": [103.0] * len(days),
        "Low": [97.0] * len(days), "Close": [100.0] * len(days),
        "Volume": [3_000_000] * len(days),
    }, index=days)


class ProvisionalUniverseTests(unittest.TestCase):
    def test_evening_universe_scan_accepts_today_bar_and_feeds_following_screen(self):
        as_of = date(2026, 11, 16)
        ticker = "1111.T"
        daily = universe_prices(as_of)
        issues = pd.DataFrame([{
            "ticker": ticker, "name": "Fixture", "market": "Prime",
        }])
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "universe.csv"
            output_path = Path(tmp) / "tickers.csv"
            with patch.object(ds, "today_jst", return_value=as_of), \
                 patch.object(ds, "is_after_tse_close", return_value=True), \
                 patch.object(ds, "previous_session", return_value=date(2026, 11, 13)), \
                 patch.object(ds, "load_jpx_list", return_value=issues), \
                 patch.object(ds.DataLoader, "fetch_all", return_value={ticker: daily}):
                ds.build_universe(
                    jpx_file="fixture.csv", cache_dir=str(Path(tmp) / "cache"),
                    state_file=str(state_path), output_file=str(output_path), batch_size=1,
                )

            state = pd.read_csv(state_path, dtype={"ticker": str})
            self.assertEqual(state.loc[0, "last_bar"], as_of.isoformat())
            self.assertGreater(state.loc[0, "avg_turnover20"], 0)
            meta = issues.iloc[0].to_dict()
            with patch.object(ds, "today_jst", return_value=as_of):
                self.assertIsNotNone(ds.compute_universe_metrics(ticker, daily, meta, 1_000_000))
                stale = universe_prices(date(2026, 11, 13))
                self.assertIsNone(ds.compute_universe_metrics(ticker, stale, meta, 1_000_000))
            tickers, _ = ds.load_provisional_tickers(state_path)
            self.assertEqual(tickers, [ticker])

            system = ds.TradingSystem(
                tickers, 1_000_000, .01, cache_dir=str(Path(tmp) / "cache"),
                report_path=None, log_path=None,
            )
            with patch.object(ds, "today_jst", return_value=as_of), \
                 patch.object(ds, "is_after_tse_close", return_value=True), \
                 patch.object(ds, "is_trading_day", return_value=True), \
                 patch.object(ds, "next_session", return_value=date(2026, 11, 17)), \
                 patch.object(ds.DataLoader, "fetch_all", return_value={ticker: daily}), \
                 patch.object(ds.Notifier, "notify") as notify:
                system.run(dry_run=True)
            self.assertTrue(notify.called)

    def test_universe_build_keeps_prior_metrics_when_current_bar_is_missing(self):
        as_of = date(2026, 11, 16)
        ticker = "1111.T"
        issues = pd.DataFrame([{
            "ticker": ticker, "name": "Fixture", "market": "Prime",
        }])
        prior_state = pd.DataFrame([{
            "ticker": ticker, "name": "Fixture", "market": "Prime",
            "avg_turnover20": 350_000_000, "atr_pct": 4.0,
            "buy_signal_days": 5, "short_signal_days": 1,
            "data_points": 140, "last_scanned": "2026-11-13",
            "last_bar": "2026-11-13",
        }])
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "universe.csv"
            prior_state.to_csv(state_path, index=False)
            before = state_path.read_bytes()
            with patch.object(ds, "today_jst", return_value=as_of), \
                 patch.object(ds, "is_after_tse_close", return_value=True), \
                 patch.object(ds, "load_jpx_list", return_value=issues), \
                 patch.object(ds.DataLoader, "fetch_all", return_value={
                     ticker: universe_prices(date(2026, 11, 13))
                 }):
                ds.build_universe(
                    jpx_file="fixture.csv", cache_dir=str(Path(tmp) / "cache"),
                    state_file=str(state_path), output_file=str(Path(tmp) / "tickers.csv"),
                    batch_size=1,
                )
            self.assertEqual(state_path.read_bytes(), before)

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
            with patch.object(ds, "today_jst", return_value=date(2026, 10, 30)), \
                 patch.object(ds, "is_after_tse_close", return_value=True), \
                 patch.object(ds, "is_trading_day", return_value=True), \
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
