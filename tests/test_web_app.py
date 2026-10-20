import json
import tempfile
import unittest
from pathlib import Path

import web


YAML_FIXTURE = """# keep this comment
estimator:
  q_translation: 100.0
  q_rotation: 400.0
  angle_variance: 0.004
  distance_variance_factor: 0.0625
  armor_yaw_variance_base: 0.09
planning:
  # send_to_control_ms: 5.0
  yaw_offset_deg: 0.0
  pitch_offset_deg: 0.0
  fallback_bullet_speed_mps: 23.0
fire:
  hit_margin_ratio: 0.6
  min_yaw_tolerance_deg: 0.5
  min_pitch_tolerance_deg: 0.5
debug:
  overlay: false
  web_frame_every: 1
  force_work_mode: ""
"""


class WebAppTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        web.CMD_PATH = str(root / "cmd.json")
        web.DATA_PATH = str(root / "data.json")
        web.LOG_PATH = str(root / "log.json")
        web.YAML_PATH = str(root / "auto_aim.yaml")
        web.RECORD_ROOT = str(root / "records")
        Path(web.YAML_PATH).write_text(YAML_FIXTURE, encoding="utf-8")
        Path(web.DATA_PATH).write_text(
            json.dumps({"time": list(range(20)), "aim/yaw": list(range(20))}),
            encoding="utf-8",
        )
        Path(web.LOG_PATH).write_text(json.dumps({"boot": {"ok": True}}), encoding="utf-8")
        web._json_cache.clear()
        web.app.config.update(TESTING=True)
        self.client = web.app.test_client()

    def tearDown(self):
        web._json_cache.clear()
        self.temp.cleanup()

    def test_index_is_offline_safe(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertNotIn("cdn.jsdelivr.net", html)
        self.assertIn("video-status", html)
        self.assertIn('id="cameraFps"', html)
        self.assertIn('id="algorithmFps"', html)
        self.assertIn('id="webFps"', html)

    def test_data_is_trimmed(self):
        data = self.client.get("/data?max_points=10").get_json()
        self.assertEqual(data["time"], list(range(10, 20)))
        self.assertEqual(data["aim/yaw"], list(range(10, 20)))

    def test_params_validate_merge_and_clear(self):
        initial = self.client.get("/params").get_json()
        keys = {item["key"] for item in initial["catalog"]}
        self.assertIn("debug.overlay", keys)
        self.assertIn("debug.web_frame_every", keys)

        response = self.client.post(
            "/params",
            json={"plan.yaw_offset_deg": 1.5, "debug.force_work_mode": "outpost"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            json.loads(Path(web.CMD_PATH).read_text(encoding="utf-8"))["debug"]["force_work_mode"],
            "outpost",
        )
        # 先读一次制造缓存，再立即追加字段；合并不能丢失上一条覆盖。
        self.client.get("/params")
        response = self.client.post("/params", json={"debug.overlay": True})
        self.assertEqual(response.status_code, 200)
        command = json.loads(Path(web.CMD_PATH).read_text(encoding="utf-8"))
        self.assertEqual(command["plan"]["yaw_offset_deg"], 1.5)
        self.assertTrue(command["debug"]["overlay"])

        before = Path(web.CMD_PATH).read_text(encoding="utf-8")
        self.assertEqual(self.client.post("/params", json={"fire.shoot_enable": True}).status_code, 400)
        self.assertEqual(self.client.post("/params", json={"camera.gain": 99}).status_code, 400)
        self.assertEqual(Path(web.CMD_PATH).read_text(encoding="utf-8"), before)

        self.assertEqual(self.client.post("/params/clear").status_code, 200)
        self.assertEqual(json.loads(Path(web.CMD_PATH).read_text(encoding="utf-8")), {})

    def test_yaml_save_preserves_comments_and_rejects_runtime_only(self):
        response = self.client.post(
            "/params/save",
            json={"plan.send_to_control_ms": 8, "debug.overlay": True},
        )
        self.assertEqual(response.status_code, 200)
        text = Path(web.YAML_PATH).read_text(encoding="utf-8")
        self.assertIn("# keep this comment", text)
        self.assertIn("send_to_control_ms: 8.0", text)
        self.assertIn("overlay: true", text)
        self.assertEqual(
            self.client.post("/params/save", json={"camera.gain": 12}).status_code,
            400,
        )

    def test_replay_series_is_sampled(self):
        session = Path(web.RECORD_ROOT) / "run_01"
        session.mkdir(parents=True)
        with (session / "series.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(250):
                handle.write(json.dumps({"t": index * 0.01, "aim": {"yaw": index}}) + "\n")
        data = self.client.get("/replays/run_01/series?max_points=100").get_json()
        self.assertEqual(data["raw_count"], 250)
        self.assertEqual(data["returned_count"], 100)
        self.assertTrue(data["sampled"])
        self.assertEqual(len(data["series"]["time"]), 100)


if __name__ == "__main__":
    unittest.main()
