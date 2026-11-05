"""Offline regression tests for independent all-signal research accounting."""
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import daytrade_screener as ds
from research_journal import ResearchSignalJournal, render_research_stats_html


def candidate(ticker, code, side, *, allocated=False, entry=100.0, stop=None):
    return {
        "ticker": ticker,
        "name": "Fixture " + ticker,
        "strategy_code": code,
        "strategy_label": "Test " + code,
        "matched_strategies": code,
        "side": side,
        "entry_price": entry,
        "stop_loss": stop if stop is not None else (95.0 if side == "buy" else 110.0),
        "allocation_status": "allocated" if allocated else "research_only",
        "allocation_reason": None if allocated else "資金配分なし",
    }


def one_bar(day, op, high, low, close):
    return pd.DataFrame(
        {"Open": [op], "High": [high], "Low": [low], "Close": [close], "Volume": [100000]},
        index=pd.DatetimeIndex([day]),
    )


def price_history():
    days = pd.bdate_range(end="2026-10-30", periods=30)
    frame = pd.DataFrame({
        "Open": [100.0] * len(days), "High": [102.0] * len(days),
        "Low": [99.0] * len(days), "Close": [100.0] * len(days),
        "Volume": [1000000] * len(days),
    }, index=days)
    frame.iloc[-1, frame.columns.get_loc("Open")] = 90.0
    frame.iloc[-1, frame.columns.get_loc("High")] = 95.0
    frame.iloc[-1, frame.columns.get_loc("Low")] = 77.0
    frame.iloc[-1, frame.columns.get_loc("Close")] = 80.0
    return frame


class ResearchJournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "research_signal_log.csv"
        self.journal = ResearchSignalJournal(self.path)

    def test_allocation_independent_records_and_idempotency(self):
        candidates = [
            candidate("1111.T", "A", "buy", allocated=True),
            candidate("2222.T", "C", "sell", allocated=False),
        ]
        df = self.journal.load()
        df, count = self.journal.append(df, candidates, "2026-11-05", "2026-11-04")
        self.assertEqual(count, 2)
        df, duplicate_count = self.journal.append(df, candidates, "2026-11-05", "2026-11-04")
        self.assertEqual(duplicate_count, 0)
        self.assertEqual(df["ticker"].tolist(), ["1111.T", "2222.T"])
        self.assertEqual(df["allocation_status"].tolist(), ["allocated", "research_only"])
        self.assertTrue((df["status"] == "pending").all())
        self.journal.save(df)
        self.assertEqual(len(self.journal.load()), 2)

    def test_buy_and_short_outcomes_and_metrics(self):
        entries = [candidate("1111.T", "A", "buy"),
                   candidate("2222.T", "C", "sell")]
        df, _ = self.journal.append(self.journal.load(), entries,
                                    "2026-11-05", "2026-11-04")
        prices = {
            "1111.T": one_bar("2026-11-05", 101.0, 109.0, 99.0, 108.0),
            "2222.T": one_bar("2026-11-05", 99.0, 104.0, 89.0, 90.0),
        }
        df, n = self.journal.resolve(df, prices, today=date(2026, 11, 6))
        self.assertEqual(n, 2)
        self.assertEqual(df["status"].tolist(), ["resolved", "resolved"])
        summary = self.journal.summarize(df)
        self.assertEqual(summary["counts"]["resolved"], 2)
        self.assertEqual(summary["overall"]["win_rate"], 100)
        self.assertGreater(summary["overall"]["avg_pnl_per_share"], 0)
        self.assertGreater(summary["by_strategy"]["A"]["avg_r"], 0)
        self.assertGreater(summary["by_strategy"]["C"]["avg_r"], 0)
        html = render_research_stats_html(summary)
        self.assertIn("A: 逆張り買い", html)
        self.assertIn("C: 急騰株空売り", html)
        self.assertIn("期待値", html)
        self.assertIn("平均損益", html)

    def test_gap_skip_is_counted_but_not_as_performance(self):
        df, _ = self.journal.append(self.journal.load(),
                                    [candidate("1111.T", "B", "buy")],
                                    "2026-11-05", "2026-11-04")
        df, n = self.journal.resolve(
            df, {"1111.T": one_bar("2026-11-05", 110.0, 112.0, 105.0, 108.0)},
            today=date(2026, 11, 6))
        self.assertEqual(n, 1)
        self.assertEqual(df.iloc[0]["status"], "skipped")
        self.assertIn("ギャップ", df.iloc[0]["skip_reason"])
        stats = self.journal.summarize(df)
        self.assertEqual(stats["counts"], {
            "total": 1, "pending": 0, "resolved": 0, "skipped": 1})
        self.assertIsNone(stats["overall"])

    def test_no_same_day_lookahead_or_missing_price_overwrite(self):
        df, _ = self.journal.append(self.journal.load(),
                                    [candidate("1111.T", "D", "buy")],
                                    "2026-11-05", "2026-11-04")
        df, n = self.journal.resolve(
            df, {"1111.T": one_bar("2026-11-05", 100.0, 110.0, 98.0, 108.0)},
            today=date(2026, 11, 5))
        self.assertEqual(n, 0)
        self.assertEqual(df.iloc[0]["status"], "pending")
        df, n = self.journal.resolve(df, {}, today=date(2026, 11, 6))
        self.assertEqual(n, 0)

    def test_loss_and_stop_are_included_in_expectancy(self):
        df, _ = self.journal.append(self.journal.load(),
                                    [candidate("1111.T", "A", "buy")],
                                    "2026-11-05", "2026-11-04")
        df, n = self.journal.resolve(
            df, {"1111.T": one_bar("2026-11-05", 101, 103, 94, 102)},
            today=date(2026, 11, 6))
        self.assertEqual(n, 1)
        self.assertTrue(df.iloc[0]["stopped_out"])
        self.assertLess(float(df.iloc[0]["r_multiple"]), 0)
        self.assertEqual(self.journal.summarize(df)["overall"]["win_rate"], 0)

    def test_unreadable_existing_csv_is_preserved(self):
        self.path.write_text("broken,columns\n1,2\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.journal.load()
        self.assertIn("broken,columns", self.path.read_text(encoding="utf-8"))


class TradingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.research = self.dir / "research.csv"
        self.portfolio = self.dir / "portfolio.csv"
        self.html = self.dir / "index.html"

    def trading(self, capital=10000):
        return ds.TradingSystem(
            tickers=["1111.T"], capital=capital, risk_pct=0.01,
            cache_dir=str(self.dir / "cache"), report_path=str(self.html),
            log_path=str(self.portfolio), research_log_path=str(self.research),
        )

    def test_rejected_signal_recorded_without_changing_portfolio_ledger(self):
        with patch.object(ds, "is_trading_day", return_value=True), \
             patch.object(ds.DataLoader, "fetch_all",
                          return_value={"1111.T": price_history()}), \
             patch.object(ds.Notifier, "notify"):
            allocated = self.trading().run(dry_run=True)
        self.assertEqual(allocated, [])
        df = ResearchSignalJournal(self.research).load()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["allocation_status"], "research_only")
        self.assertEqual(df.iloc[0]["status"], "pending")
        portfolio = pd.read_csv(self.portfolio)
        self.assertEqual(len(portfolio), 0)
        page = self.html.read_text(encoding="utf-8")
        self.assertIn("全候補の研究用・仮想検証", page)
        self.assertIn("資金配分候補の仮想検証", page)

    def test_pending_research_ticker_is_fetched_even_if_no_longer_selected(self):
        journal = ResearchSignalJournal(self.research)
        prior, _ = journal.append(
            journal.load(), [candidate("2222.T", "A", "buy")],
            "2026-10-30", "2026-10-29")
        journal.save(prior)
        captured = []

        def fetch_prices(loader):
            captured.extend(loader.tickers)
            return {
                "1111.T": price_history(),
                "2222.T": one_bar("2026-10-30", 101, 108, 99, 107),
            }

        with patch.object(ds, "is_trading_day", return_value=True), \
             patch.object(ds.DataLoader, "fetch_all", fetch_prices), \
             patch.object(ds.Notifier, "notify"):
            self.trading().run(dry_run=True)
        self.assertIn("2222.T", captured)
        df = journal.load()
        old = df.loc[df["ticker"] == "2222.T"].iloc[0]
        self.assertEqual(old["status"], "resolved")
        self.assertGreater(float(old["r_multiple"]), 0)

    def test_low_coverage_does_not_overwrite_prior_research_log(self):
        journal = ResearchSignalJournal(self.research)
        prior, _ = journal.append(
            journal.load(), [candidate("2222.T", "A", "buy")],
            "2026-10-30", "2026-10-29")
        journal.save(prior)
        original = self.research.read_bytes()
        with patch.object(ds, "is_trading_day", return_value=True), \
             patch.object(ds.DataLoader, "fetch_all", return_value={}):
            self.assertEqual(self.trading().run(dry_run=True), [])
        self.assertEqual(self.research.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
