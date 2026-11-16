"""Offline regression tests for evening screening and session-date handling."""
import sys
import tempfile
import types
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import daytrade_screener as ds
from research_journal import ResearchSignalJournal


AS_OF = date(2026, 11, 16)  # Monday
NEXT_DAY = date(2026, 11, 17)


def bars(days):
    return pd.DataFrame({
        "Open": [100.0] * len(days), "High": [102.0] * len(days),
        "Low": [99.0] * len(days), "Close": [100.0] * len(days),
        "Volume": [100000] * len(days),
    }, index=pd.DatetimeIndex(days))


def calendar_module(holidays=()):
    holiday_set = set(holidays)

    class TokyoCalendar:
        @staticmethod
        def is_session(ts):
            day = ts.date()
            return day.weekday() < 5 and day not in holiday_set

    return types.SimpleNamespace(get_calendar=lambda name: TokyoCalendar())


class EveningScheduleTests(unittest.TestCase):
    def test_tse_close_gate_is_false_before_1530_and_true_at_close(self):
        self.assertFalse(ds.is_after_tse_close(datetime(2026, 11, 16, 15, 29, 59)))
        self.assertTrue(ds.is_after_tse_close(datetime(2026, 11, 16, 15, 30, 0)))

    def test_next_session_regular_monday(self):
        with patch.dict(sys.modules, {"exchange_calendars": calendar_module()}):
            self.assertEqual(ds.next_session(AS_OF), NEXT_DAY)

    def test_friday_targets_monday(self):
        friday = date(2026, 11, 13)
        with patch.dict(sys.modules, {"exchange_calendars": calendar_module()}):
            self.assertEqual(ds.next_session(friday), date(2026, 11, 16))

    def test_holiday_after_friday_targets_tuesday(self):
        friday = date(2026, 11, 20)
        with patch.dict(sys.modules, {
            "exchange_calendars": calendar_module({date(2026, 11, 23)})
        }):
            self.assertEqual(ds.next_session(friday), date(2026, 11, 24))

    def test_year_end_uses_first_january_tse_session(self):
        holidays = {date(2026, 12, 31), date(2027, 1, 1)}
        with patch.dict(sys.modules, {
            "exchange_calendars": calendar_module(holidays)
        }):
            self.assertEqual(ds.next_session(date(2026, 12, 30)), date(2027, 1, 4))

    def test_data_normalization_keeps_asof_bar_and_rejects_future(self):
        source = bars([AS_OF - timedelta(days=1), AS_OF, NEXT_DAY])
        result = ds.DataLoader.normalize(source, AS_OF)
        self.assertEqual([ts.date() for ts in result.index],
                         [AS_OF - timedelta(days=1), AS_OF])

    def test_final_bar_with_invalid_ohlc_or_zero_volume_is_rejected(self):
        self.assertFalse(ds.DataLoader._valid_bar(pd.Series({
            "Open": 100, "High": 99, "Low": 98, "Close": 101, "Volume": 1000,
        })))
        self.assertFalse(ds.DataLoader._valid_bar(pd.Series({
            "Open": 100, "High": 102, "Low": 98, "Close": 101, "Volume": 0,
        })))

    def test_yfinance_download_end_is_exclusive_day_after_asof(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader = ds.DataLoader(["1111.T"], cache_dir=tmp, as_of_date=AS_OF)
            with patch.object(ds.yf, "download", return_value=pd.DataFrame()) as download:
                loader._download(["1111.T"], start="2026-11-01")
            self.assertEqual(download.call_args.kwargs["end"], "2026-11-17")

    def test_report_identifies_price_date_and_target_session(self):
        html = ds.ReportGenerator().build_html(
            [], {"ad_ratio": None, "avg_dev5": 0}, capital=100000,
            risk_pct=.01, as_of_date=AS_OF, signal_date=NEXT_DAY)
        self.assertIn("判定に使用した確定日足: 2026-11-16", html)
        self.assertIn("売買候補の対象営業日: 2026-11-17", html)
        self.assertIn("翌東証営業日のデイトレ候補", html)

    def test_cache_with_only_previous_session_is_not_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader = ds.DataLoader(["1111.T"], cache_dir=tmp, as_of_date=AS_OF,
                                   max_retries=1, retry_wait_sec=0)
            previous = bars(pd.bdate_range(end="2026-11-13", periods=20))
            loader._save_cache("1111.T", previous)
            with patch.object(ds, "today_jst", return_value=AS_OF), \
                 patch.object(ds, "is_after_tse_close", return_value=True), \
                 patch.object(loader, "_download", return_value=pd.DataFrame()):
                result = loader.fetch_all()
            self.assertEqual(result, {})
            self.assertEqual(loader.failed, ["1111.T"])
            cached = loader._read_cache("1111.T")
            self.assertEqual(cached.index[-1].date(), date(2026, 11, 13))

    def test_current_day_cache_is_refetched_before_it_can_be_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader = ds.DataLoader(["1111.T"], cache_dir=tmp, as_of_date=AS_OF,
                                   max_retries=1, retry_wait_sec=0)
            cached = bars(pd.bdate_range(end=AS_OF, periods=20))
            cached.iloc[-1, cached.columns.get_loc("Close")] = 100
            loader._save_cache("1111.T", cached)
            fresh = cached.copy()
            fresh.iloc[-1, fresh.columns.get_loc("Close")] = 101
            with patch.object(ds, "today_jst", return_value=AS_OF), \
                 patch.object(ds, "is_after_tse_close", return_value=True), \
                 patch.object(loader, "_download", return_value=fresh) as download:
                result = loader.fetch_all()
            self.assertEqual(download.call_count, 1)
            self.assertEqual(result["1111.T"].iloc[-1]["Close"], 101)

    def test_failed_refresh_does_not_fall_back_to_current_day_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader = ds.DataLoader(["1111.T"], cache_dir=tmp, as_of_date=AS_OF,
                                   max_retries=1, retry_wait_sec=0)
            cached = bars(pd.bdate_range(end=AS_OF, periods=20))
            loader._save_cache("1111.T", cached)
            cache_path = loader._cache_path("1111.T")
            before = cache_path.read_bytes()
            with patch.object(ds, "today_jst", return_value=AS_OF), \
                 patch.object(ds, "is_after_tse_close", return_value=True), \
                 patch.object(loader, "_download", return_value=pd.DataFrame()) as download:
                result = loader.fetch_all()
            self.assertEqual(download.call_count, 2)  # batch and individual retry
            self.assertEqual(result, {})
            self.assertEqual(loader.failed, ["1111.T"])
            self.assertEqual(cache_path.read_bytes(), before)

    def test_preclose_loader_does_not_use_current_day_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader = ds.DataLoader(["1111.T"], cache_dir=tmp, as_of_date=AS_OF)
            loader._save_cache("1111.T", bars(pd.bdate_range(end=AS_OF, periods=20)))
            with patch.object(ds, "today_jst", return_value=AS_OF), \
                 patch.object(ds, "is_after_tse_close", return_value=False), \
                 patch.object(loader, "_download") as download:
                self.assertEqual(loader.fetch_all(), {})
            self.assertFalse(download.called)
            self.assertEqual(loader.failed, ["1111.T"])

    def test_preclose_blocks_screening_and_universe_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "signals.csv"
            log.write_text("preserve this", encoding="utf-8")
            system = ds.TradingSystem(
                ["1111.T"], 1000000, .01, cache_dir=str(Path(tmp) / "cache"),
                report_path=str(Path(tmp) / "report.html"), log_path=str(log),
            )
            with patch.object(ds, "is_after_tse_close", return_value=False), \
                 patch.object(ds.DataLoader, "fetch_all") as fetch, \
                 patch.object(ds.Notifier, "notify") as notify:
                self.assertEqual(system.run(), [])
                self.assertTrue(ds.build_universe(
                    jpx_file="unused.csv", state_file=str(Path(tmp) / "universe.csv"),
                    output_file=str(Path(tmp) / "tickers.csv"),
                ).empty)
            self.assertFalse(fetch.called)
            self.assertFalse(notify.called)
            self.assertEqual(log.read_text(encoding="utf-8"), "preserve this")

    def test_run_stops_on_missing_asof_bar_without_touching_existing_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "signals.csv"
            research = Path(tmp) / "research.csv"
            signals = ds.SignalLogger(log)
            signal_row = {
                "ticker": "2222.T", "name": "Old", "strategy_code": "A",
                "strategy_label": "A", "side": "buy", "entry_price": 100,
                "stop_loss": 95, "max_shares": 100, "estimated_cost": 10000,
                "risk_amount": 500,
            }
            old_signal_log = signals.append_new(
                signals.load(), [signal_row], {}, "2026-11-13", "2026-11-12")
            old_signal_log.loc[0, "status"] = "resolved"
            old_signal_log.loc[0, "r_multiple"] = 1.25
            signals.save(old_signal_log)
            journal = ResearchSignalJournal(research)
            old_research, _ = journal.append(journal.load(), [{
                "ticker": "2222.T", "name": "Old", "strategy_code": "A",
                "strategy_label": "A", "side": "buy", "entry_price": 100,
                "stop_loss": 95,
            }], "2026-11-13", "2026-11-12")
            old_research.loc[0, "status"] = "resolved"
            old_research.loc[0, "r_multiple"] = 0.75
            journal.save(old_research)
            before = (log.read_bytes(), research.read_bytes())
            system = ds.TradingSystem(
                ["1111.T"], 1000000, .01, cache_dir=str(Path(tmp) / "cache"),
                report_path=None, log_path=str(log), research_log_path=str(research),
            )
            stale = bars(pd.bdate_range(end="2026-11-13", periods=20))
            with patch.object(ds, "today_jst", return_value=AS_OF), \
                 patch.object(ds, "is_after_tse_close", return_value=True), \
                 patch.object(ds, "is_trading_day", return_value=True), \
                 patch.object(ds.DataLoader, "fetch_all", return_value={"1111.T": stale}), \
                 patch.object(ds.Notifier, "notify") as notify:
                self.assertEqual(system.run(), [])
            self.assertFalse(notify.called)
            self.assertEqual((log.read_bytes(), research.read_bytes()), before)
            self.assertEqual(float(signals.load().iloc[0]["r_multiple"]), 1.25)
            self.assertEqual(float(journal.load().iloc[0]["r_multiple"]), 0.75)

    def test_repeated_evening_run_does_not_duplicate_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "signals.csv"
            research = Path(tmp) / "research.csv"
            history = bars(pd.bdate_range(end=AS_OF, periods=30))
            history.iloc[-1, history.columns.get_loc("Open")] = 90
            history.iloc[-1, history.columns.get_loc("High")] = 95
            history.iloc[-1, history.columns.get_loc("Low")] = 77
            history.iloc[-1, history.columns.get_loc("Close")] = 80
            system = ds.TradingSystem(
                ["1111.T"], 1000000, .01, cache_dir=str(Path(tmp) / "cache"),
                report_path=None, log_path=str(log), research_log_path=str(research),
            )
            with patch.object(ds, "today_jst", return_value=AS_OF), \
                 patch.object(ds, "is_after_tse_close", return_value=True), \
                 patch.object(ds, "is_trading_day", return_value=True), \
                 patch.object(ds.DataLoader, "fetch_all", return_value={"1111.T": history}), \
                 patch.object(ds.Notifier, "notify") as notify:
                system.run(dry_run=True)
                system.run(dry_run=True)
            self.assertEqual(len(pd.read_csv(log)), 1)
            research_frame = ResearchSignalJournal(research).load()
            self.assertEqual(len(research_frame), 1)
            message = ds.Notifier.format_short_message(
                [], {"ad_ratio": None, "avg_dev5": 0},
                as_of_date=AS_OF, signal_date=NEXT_DAY)
            self.assertIn("対象: 2026-11-17", message)
            self.assertIn("確定日足: 2026-11-16", message)
            displayed = notify.call_args.args[0]
            self.assertIsInstance(displayed, list)

    def test_same_session_signal_can_be_resolved_only_with_asof_confirmation(self):
        journal = ResearchSignalJournal("unused.csv")
        entry = {
            "ticker": "1111.T", "name": "Fixture", "strategy_code": "A",
            "strategy_label": "A", "matched_strategies": "A", "side": "buy",
            "entry_price": 100, "stop_loss": 95, "allocation_status": "research_only",
        }
        pending, _ = journal.append(journal.load(), [entry], AS_OF.isoformat(),
                                    date(2026, 11, 13).isoformat())
        same_day_prices = {"1111.T": bars([AS_OF])}
        unchanged, count = journal.resolve(pending, same_day_prices, as_of_date=AS_OF)
        self.assertEqual(count, 1)
        self.assertEqual(unchanged.iloc[0]["status"], "resolved")

        still_pending, count = journal.resolve(pending, same_day_prices, today=AS_OF)
        self.assertEqual(count, 0)
        self.assertEqual(still_pending.iloc[0]["status"], "pending")

    def test_signal_logger_same_day_resolution_requires_asof_date(self):
        logger = ds.SignalLogger("unused.csv")
        signal = {
            "ticker": "1111.T", "name": "Fixture", "strategy_code": "A",
            "strategy_label": "A", "matched_strategies": "A", "side": "buy",
            "entry_price": 100, "stop_loss": 95, "max_shares": 100,
            "estimated_cost": 10000, "risk_amount": 500,
        }
        pending = logger.append_new(logger.load(), [signal], {}, AS_OF.isoformat(),
                                    date(2026, 11, 13).isoformat())
        resolved = logger.resolve(pending, {"1111.T": bars([AS_OF])}, as_of_date=AS_OF)
        self.assertIn(resolved.iloc[0]["status"], ("resolved", "skipped_gap"))
        self.assertNotEqual(resolved.iloc[0]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
