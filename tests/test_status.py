import json
import tempfile
import unittest
from pathlib import Path

from presence_guard.status import ControlClient, ControlServer, StatusStore


class StatusStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = StatusStore(self.tmp.name)

    def test_publish_and_read(self):
        self.store.add_event("服务启动", "info")
        self.store.publish(state="present", state_text="在工位", present=True,
                           badge="present", badge_text="已佩戴工牌")
        payload = StatusStore.read(self.tmp.name)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["state"], "present")
        self.assertEqual(payload["badge"], "present")
        self.assertEqual(payload["events"][0]["text"], "服务启动")

    def test_events_are_capped(self):
        for i in range(80):
            self.store.add_event(f"event {i}")
        self.store.publish(state="present")
        self.assertLessEqual(len(StatusStore.read(self.tmp.name)["events"]), 40)

    def test_read_missing_is_none(self):
        self.assertIsNone(StatusStore.read(str(Path(self.tmp.name) / "nope")))


class ControlServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls = []

        def handler(request):
            self.calls.append(request)
            if request.get("cmd") == "ping":
                return {"ok": True, "state": "present"}
            return {"ok": True, "cmd": request.get("cmd")}

        self.server = ControlServer(self.tmp.name, handler)
        self.server.start()
        self.addCleanup(self.server.stop)
        self.client = ControlClient(self.tmp.name)

    def test_ping(self):
        response = self.client.send("ping")
        self.assertTrue(response["ok"])
        self.assertEqual(response["state"], "present")

    def test_pause_payload(self):
        response = self.client.send("pause", seconds=123)
        self.assertTrue(response["ok"])
        self.assertEqual(self.calls[-1]["seconds"], 123)

    def test_unknown_command_returns_reply(self):
        response = self.client.send("does-not-exist")
        self.assertTrue(response["ok"])
        self.assertEqual(response["cmd"], "does-not-exist")


if __name__ == "__main__":
    unittest.main()
