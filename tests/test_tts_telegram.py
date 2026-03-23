import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import tts_telegram


class RequiredMemTests(unittest.TestCase):
    def test_profile_routing_uses_thresholds(self):
        profile = {
            "min_mem_available_gb": 3.0,
            "routing": {
                "short_chars": 400,
                "medium_chars": 1200,
                "long_chars": 2200,
                "short_mem_gb": 1.6,
                "medium_mem_gb": 2.2,
                "long_mem_gb": 2.8,
                "xlong_mem_gb": 3.0,
            },
        }
        self.assertEqual(tts_telegram.profile_required_mem(profile, 100, None), 1.6)
        self.assertEqual(tts_telegram.profile_required_mem(profile, 1000, None), 2.2)
        self.assertEqual(tts_telegram.profile_required_mem(profile, 1800, None), 2.8)
        self.assertEqual(tts_telegram.profile_required_mem(profile, 3000, None), 3.0)

    def test_override_wins(self):
        self.assertEqual(tts_telegram.profile_required_mem({}, 999, 4.2), 4.2)


class RouteTests(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            text="Hallo Welt",
            chat_id="123",
            reply_to=None,
            caption=None,
            speaker="sohee",
            language="german",
            server="http://127.0.0.1:5050",
            max_time=5,
            legacy_port=5052,
            tmpdir=tempfile.gettempdir(),
            keep=False,
            dry_run=False,
            bot_token="dummy",
            longtext_mem_gb=None,
        )

    @mock.patch("tts_telegram.read_bot_token", return_value="dummy")
    def test_routes_low_mem_faster_to_legacy(self, _token):
        orch = tts_telegram.Orchestrator(self._args())
        with mock.patch("tts_telegram.snapshot_service") as snap, mock.patch.object(orch.legacy, "start") as legacy_start:
            snap.return_value = tts_telegram.ServiceSnapshot(
                base_url=orch.primary_url,
                health={"status": "ok", "mem_available_gb": 1.0},
                info={
                    "engine": "faster-qwen3-tts:large",
                    "profile": {
                        "min_mem_available_gb": 3.0,
                        "routing": {"short_chars": 400, "medium_chars": 1200, "long_chars": 2200, "short_mem_gb": 1.6, "medium_mem_gb": 2.2, "long_mem_gb": 2.8, "xlong_mem_gb": 3.0},
                    },
                },
            )
            route = orch.choose_route()
        self.assertEqual(route, orch.legacy.url)
        self.assertTrue(orch.restore_primary)
        legacy_start.assert_called_once()

    @mock.patch("tts_telegram.read_bot_token", return_value="dummy")
    def test_startup_mem_error_skips_restart_loop(self, _token):
        orch = tts_telegram.Orchestrator(self._args())
        with mock.patch("tts_telegram.snapshot_service") as snap, mock.patch.object(orch, "restart_primary") as restart, mock.patch.object(orch.legacy, "start") as legacy_start:
            snap.return_value = tts_telegram.ServiceSnapshot(
                base_url=orch.primary_url,
                health={"status": "degraded", "startup_error": "Insufficient MemAvailable for faster profile"},
                info={"engine": "faster-qwen3-tts:large", "profile": {}},
            )
            route = orch.choose_route()
        self.assertEqual(route, orch.legacy.url)
        restart.assert_not_called()
        legacy_start.assert_called_once()


class CleanupTests(unittest.TestCase):
    @mock.patch("tts_telegram.read_bot_token", return_value="dummy")
    def test_keep_preserves_tempdir(self, _token):
        args = SimpleNamespace(
            text="Hallo",
            chat_id="1",
            reply_to=None,
            caption=None,
            speaker="sohee",
            language="german",
            server="http://127.0.0.1:5050",
            max_time=5,
            legacy_port=5052,
            tmpdir=tempfile.gettempdir(),
            keep=True,
            dry_run=True,
            bot_token="dummy",
            longtext_mem_gb=None,
        )
        orch = tts_telegram.Orchestrator(args)
        wav, _ = orch._temp_paths()
        wav.write_bytes(b"123")
        temp_root = Path(orch.temp_dir_path)
        orch.cleanup()
        self.assertTrue(temp_root.exists())


if __name__ == "__main__":
    unittest.main()
