"""Offline tests for retryable, LINE-only report delivery."""
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import daytrade_screener as ds


AS_OF = date(2026, 11, 16)
TARGET = date(2026, 11, 17)


def bars():
    days = pd.bdate_range(end=AS_OF, periods=35)
    return pd.DataFrame({
        "Open": [100.0] * len(days), "High": [103.0] * len(days),
        "Low": [97.0] * len(days), "Close": [100.0] * len(days),
        "Volume": [100_000] * len(days),
    }, index=days)


class DeliveryReliabilityTests(unittest.TestCase):
    def test_workflow_has_three_weekday_schedule_opportunities(self):
        workflow = (Path(__file__).parents[1] / ".github/workflows/daytrade_screener.yml").read_text(
            encoding="utf-8")
        self.assertEqual(
            ["30 8 * * 1-5", "47 8 * * 1-5", "13 9 * * 1-5"],
            [line.split("'")[1] for line in workflow.splitlines() if "cron:" in line],
        )
        self.assertIn("遅延・欠落し得る", workflow)
        self.assertIn("logs/delivery_status.json", workflow)
        self.assertIn("set -euo pipefail", workflow)
        self.assertIn("Failed to commit report or delivery state", workflow)
        self.assertIn("Failed to push report or delivery state", workflow)

    def test_delivery_ledger_first_run_and_atomic_persistent_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logs" / "delivery_status.json"
            ledger = ds.DeliveryLedger(path)
            self.assertFalse(ledger.line_sent(TARGET))
            ledger.mark_line_sent(TARGET)
            self.assertTrue(ledger.line_sent(TARGET))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {
                "version": 1, "deliveries": {"2026-11-17": {"line": "sent"}},
            })

    def test_failed_line_retries_then_success_is_not_sent_again_and_discord_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "logs" / "delivery_status.json"
            system = ds.TradingSystem(
                ["1111.T"], 1_000_000, .01, cache_dir=str(Path(tmp) / "cache"),
                report_path=None, log_path=None, research_log_path=None,
                delivery_state_path=str(ledger_path),
            )
            with patch.object(ds, "today_jst", return_value=AS_OF), \
                 patch.object(ds, "next_session", return_value=TARGET), \
                 patch.object(ds, "is_after_tse_close", return_value=True), \
                 patch.object(ds, "is_trading_day", return_value=True), \
                 patch.object(ds.DataLoader, "fetch_all", return_value={"1111.T": bars()}), \
                 patch.dict("os.environ", {
                     "LINE_CHANNEL_ACCESS_TOKEN": "fixture-token",
                     "LINE_USER_ID": "fixture-user",
                     "DISCORD_WEBHOOK_URL": "https://example.invalid/webhook",
                 }), \
                 patch.object(ds.Notifier, "send_line", side_effect=[False, True]) as send_line, \
                 patch.object(ds.Notifier, "send_discord") as send_discord:
                system.run()
                self.assertFalse(ledger_path.exists())
                system.run()
                self.assertTrue(ds.DeliveryLedger(ledger_path).line_sent(TARGET))
                system.run()

            self.assertEqual(send_line.call_count, 2)
            self.assertFalse(send_discord.called)

    def test_successful_line_send_with_state_save_failure_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            system = ds.TradingSystem(
                ["1111.T"], 1_000_000, .01, cache_dir=str(Path(tmp) / "cache"),
                report_path=None, log_path=None, research_log_path=None,
                delivery_state_path=str(Path(tmp) / "delivery.json"),
            )
            with patch.object(ds, "today_jst", return_value=AS_OF), \
                 patch.object(ds, "next_session", return_value=TARGET), \
                 patch.object(ds, "is_after_tse_close", return_value=True), \
                 patch.object(ds, "is_trading_day", return_value=True), \
                 patch.object(ds.DataLoader, "fetch_all", return_value={"1111.T": bars()}), \
                 patch.dict("os.environ", {
                     "LINE_CHANNEL_ACCESS_TOKEN": "fixture-token",
                     "LINE_USER_ID": "fixture-user",
                 }), \
                 patch.object(ds.Notifier, "send_line", return_value=True) as send_line, \
                 patch.object(ds.DeliveryLedger, "mark_line_sent",
                              side_effect=OSError("fixture write failure")):
                with self.assertRaisesRegex(RuntimeError, "配信済み記録保存に失敗"):
                    system.run()
            send_line.assert_called_once()

    def test_existing_success_record_skips_before_fetch_and_notification(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "delivery.json"
            ledger = ds.DeliveryLedger(ledger_path)
            ledger.mark_line_sent(TARGET)
            system = ds.TradingSystem(
                ["1111.T"], 1_000_000, .01, cache_dir=str(Path(tmp) / "cache"),
                report_path=None, log_path=None, research_log_path=None,
                delivery_state_path=str(ledger_path),
            )
            with patch.object(ds, "today_jst", return_value=AS_OF), \
                 patch.object(ds, "next_session", return_value=TARGET), \
                 patch.object(ds, "is_after_tse_close", return_value=True), \
                 patch.object(ds, "is_trading_day", return_value=True), \
                 patch.object(ds.DataLoader, "fetch_all") as fetch, \
                 patch.object(ds.Notifier, "notify") as notify:
                self.assertEqual(system.run(), [])
            self.assertFalse(fetch.called)
            self.assertFalse(notify.called)

    def test_discord_never_sends_and_line_success_is_the_only_success_result(self):
        notifier = ds.Notifier(
            discord_webhook_url="https://example.invalid/webhook",
            line_channel_access_token="fixture-token", line_user_id="fixture-user",
        )
        with patch.object(ds.Notifier, "send_line", return_value=True) as send_line, \
             patch.object(ds.requests, "post") as post:
            result = notifier.notify([], {"ad_ratio": None, "avg_dev5": 0},
                                     as_of_date=AS_OF, signal_date=TARGET)
        self.assertTrue(result)
        send_line.assert_called_once()
        post.assert_not_called()

    def test_missing_line_credentials_skip_without_marking_success(self):
        notifier = ds.Notifier(discord_webhook_url="https://example.invalid/webhook")
        with patch.object(ds.Notifier, "send_line") as send_line:
            result = notifier.notify([], {"ad_ratio": None, "avg_dev5": 0},
                                     as_of_date=AS_OF, signal_date=TARGET)
        self.assertFalse(result)
        self.assertFalse(send_line.called)

    def test_report_generation_failure_does_not_attempt_notification(self):
        with tempfile.TemporaryDirectory() as tmp:
            system = ds.TradingSystem(
                ["1111.T"], 1_000_000, .01, cache_dir=str(Path(tmp) / "cache"),
                report_path=str(Path(tmp) / "report.html"), log_path=None,
                research_log_path=None, delivery_state_path=str(Path(tmp) / "delivery.json"),
            )
            with patch.object(ds, "today_jst", return_value=AS_OF), \
                 patch.object(ds, "next_session", return_value=TARGET), \
                 patch.object(ds, "is_after_tse_close", return_value=True), \
                 patch.object(ds, "is_trading_day", return_value=True), \
                 patch.object(ds.DataLoader, "fetch_all", return_value={"1111.T": bars()}), \
                 patch.object(system.report_generator, "build_html", side_effect=RuntimeError("fixture")), \
                 patch.object(ds.Notifier, "notify") as notify:
                self.assertEqual(system.run(), [])
            self.assertFalse(notify.called)
            self.assertFalse(Path(tmp, "delivery.json").exists())


if __name__ == "__main__":
    unittest.main()
