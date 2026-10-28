#!/usr/bin/env python3
"""FSVISION 远程 Web 调试器。

只读 /dev/shm/newvision_* 三个文件，不碰瞄准链路：
  newvision_frame     uint32 LE 长度 + JPEG 标注帧
  newvision_data.json 时序（telemetryFrame 拍平，含 time 与 "gimbal/yaw" 等）
  newvision_log.json  结构化日志

启动:
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
  .venv/bin/python web.py          # 0.0.0.0:9000
"""

from __future__ import annotations

import json
import logging
import math
import mmap
import os
import re
import socket
import struct
import threading
import time
from collections import deque

from flask import Flask, Response, jsonify, render_template, request


def bounded_env_int(name: str, default: int, lower: int, upper: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, lower), upper)


IPC_PREFIX = os.environ.get("VISION_TOOLS_IPC_PREFIX", "/dev/shm/newvision")
FRAME_PATH = IPC_PREFIX + "_frame"
DATA_PATH = IPC_PREFIX + "_data.json"
LOG_PATH = IPC_PREFIX + "_log.json"
CMD_PATH = IPC_PREFIX + "_cmd.json"
YAML_PATH = os.environ.get(
    "VISION_TOOLS_YAML", os.environ.get("NEWVISION_YAML", "config/auto_aim.yaml")
)
FRAME_MAP_SIZE = 2 * 1024 * 1024
PORT = bounded_env_int("FSVISION_PORT", 9000, 1, 65535)
LOGGER_WS_PORT = int(os.environ.get("NEWVISION_LOG_WS_PORT", "1234"))
DEFAULT_VIDEO_FPS = 30
MAX_VIDEO_FPS = 30
# 实时档跟随 WebSink 实际产生的新帧；轮询频率只决定发现新帧的最快速度。
FRAME_READ_FPS = bounded_env_int("FSVISION_SOURCE_POLL_FPS", 60, 30, 120)
JSON_CACHE_TTL = 0.05

app = Flask(__name__, template_folder="templates", static_folder="static")
# 调试台经常与后端同时更新；禁用静态长缓存，避免旧 JS 配新 HTML。
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

_latest_jpeg: bytes | None = None
_latest_seq = 0
_frame_cond = threading.Condition()
_stop = threading.Event()
_use_shm = False
_mapfile: mmap.mmap | None = None
_fd: int | None = None

_json_cache: dict[str, dict] = {}
_json_lock = threading.Lock()
_logger_recent = deque(maxlen=500)


def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return "127.0.0.1"


def init_shared_memory(verbose: bool = True) -> bool:
    global _use_shm, _mapfile, _fd

    if _mapfile is not None:
        try:
            _mapfile.close()
        except Exception:
            pass
        _mapfile = None
    if _fd is not None:
        try:
            os.close(_fd)
        except Exception:
            pass
        _fd = None

    try:
        _fd = os.open(FRAME_PATH, os.O_RDONLY)
        _mapfile = mmap.mmap(_fd, FRAME_MAP_SIZE, access=mmap.ACCESS_READ)
        _use_shm = True
        if verbose:
            print("shared memory ready:", FRAME_PATH)
        return True
    except OSError as error:
        _use_shm = False
        if verbose:
            print("shared memory unavailable:", error)
        return False


def read_jpeg_from_shared_memory() -> bytes | None:
    if not (_use_shm and _mapfile):
        return None
    try:
        _mapfile.seek(0)
        size_bytes = _mapfile.read(4)
        if len(size_bytes) != 4:
            return None
        size = struct.unpack("<I", size_bytes)[0]
        if not (0 < size <= FRAME_MAP_SIZE - 4):
            return None
        data = _mapfile.read(size)
        if len(data) != size or not data.startswith(b"\xff\xd8"):
            return None
        # WebSink 发布新图时会先把长度清零；前后长度不一致说明恰好撞上写入，
        # 丢掉这一轮即可，不能把半张 JPEG 推给浏览器。
        _mapfile.seek(0)
        if _mapfile.read(4) != size_bytes:
            return None
        return data
    except Exception:
        return None


def read_json_cached(path: str):
    now = time.monotonic()
    try:
        stat = os.stat(path)
    except OSError as error:
        raise FileNotFoundError(path) from error

    with _json_lock:
        cached = _json_cache.get(path)
        if cached and cached["mtime_ns"] == stat.st_mtime_ns:
            cached["checked_at"] = now
            return cached["data"]
        if cached and now - cached["checked_at"] < JSON_CACHE_TTL:
            return cached["data"]

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    with _json_lock:
        _json_cache[path] = {
            "mtime_ns": stat.st_mtime_ns,
            "checked_at": now,
            "data": data,
        }
    return data


def no_store_json(data, status=200):
    response = jsonify(data)
    response.status_code = status
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


def clamp_int(value, default, lower, upper):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, lower), upper)


def trim_series_json(data: dict, max_points: int) -> dict:
    if not isinstance(data, dict) or max_points <= 0:
        return data
    time_values = data.get("time")
    if not isinstance(time_values, list) or len(time_values) <= max_points:
        return data
    start = len(time_values) - max_points
    trimmed = {}
    for key, value in data.items():
        if isinstance(value, list) and len(value) == len(time_values):
            trimmed[key] = value[start:]
        else:
            trimmed[key] = value
    return trimmed


def frame_reader_loop() -> None:
    global _latest_jpeg, _latest_seq, _use_shm

    last_retry = 0.0
    last_log = 0.0
    interval = 1.0 / FRAME_READ_FPS

    while not _stop.is_set():
        jpeg = read_jpeg_from_shared_memory()
        if jpeg is not None:
            with _frame_cond:
                # 共享内存没有序号；比较 JPEG 内容以免静止时重复推送同一帧。
                if jpeg != _latest_jpeg:
                    _latest_jpeg = jpeg
                    _latest_seq += 1
                    _frame_cond.notify_all()
            time.sleep(interval)
            continue

        now = time.time()
        if now - last_retry > 5.0:
            verbose = now - last_log > 30.0
            if verbose:
                print("retrying shared memory...")
                last_log = now
            init_shared_memory(verbose=verbose)
            last_retry = now
        time.sleep(0.2)


def mjpeg_stream(max_fps: int | None):
    seen_seq = -1
    last_sent_at: float | None = None
    min_interval = 0.0 if max_fps is None else 1.0 / max_fps
    while not _stop.is_set():
        with _frame_cond:
            _frame_cond.wait_for(
                lambda: _latest_seq != seen_seq or _stop.is_set(),
                timeout=1.0,
            )
            if _stop.is_set():
                break
            frame = _latest_jpeg
            frame_seq = _latest_seq
        if frame is None:
            continue

        # 限速期间让源继续更新，到发送时再取最新帧。旧实现先把序号标成
        # “已看”再直接丢弃，40 FPS 输入会隔帧变成约 20 FPS。
        if min_interval and last_sent_at is not None:
            remaining = last_sent_at + min_interval - time.monotonic()
            if remaining > 0.0 and _stop.wait(remaining):
                break
            with _frame_cond:
                frame = _latest_jpeg
                frame_seq = _latest_seq
        if frame is None:
            continue
        seen_seq = frame_seq
        content_length = str(len(frame)).encode("ascii")
        last_sent_at = time.monotonic()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + content_length + b"\r\n\r\n" + frame + b"\r\n"
        )


@app.route("/")
def index():
    return render_template(
        "index.html",
        server_url=f"http://{local_ip()}:{PORT}",
        logger_ws_port=LOGGER_WS_PORT,
        video_fps=DEFAULT_VIDEO_FPS,
    )


@app.route("/video")
def video_feed():
    requested_fps = request.args.get("fps", str(DEFAULT_VIDEO_FPS))
    realtime = str(requested_fps).lower() in {"0", "realtime"}
    fps = None if realtime else clamp_int(requested_fps, DEFAULT_VIDEO_FPS, 1, MAX_VIDEO_FPS)
    response = Response(
        mjpeg_stream(fps),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["X-Stream-FPS"] = "realtime" if realtime else str(fps)
    return response


@app.route("/data")
def get_data():
    try:
        max_points = clamp_int(request.args.get("max_points"), 200, 10, 1000)
        data = trim_series_json(read_json_cached(DATA_PATH), max_points)
        return no_store_json(data)
    except Exception as error:  # noqa: BLE001 — 前端状态灯需要错误正文
        return no_store_json({"error": str(error)}, 500)


@app.route("/log")
def get_log():
    try:
        return no_store_json(read_json_cached(LOG_PATH))
    except Exception as error:  # noqa: BLE001
        return no_store_json({"error": str(error)}, 500)


# ---- 在线调参：运行时热更走 cmd 通道，持久化单独写 YAML ----

# 白名单：与 runtime::applyWebTuning 对齐。min/max 同时用于 UI 和 HTTP 入口校验。
# yaml_key = config/auto_aim.yaml 里的**字面**键名（可能与 web key 不同后缀）。
TUNABLE_PARAMS = [
    {"key": "plan.send_to_control_ms", "yaml_key": "send_to_control_ms",
     "label": "send_to_control", "name": "控制链路延迟", "unit": "ms",
     "description": "规划完成到下位机开始执行的补偿时间。过小会跟随滞后，过大会预测超前；应对照 aim/yaw 与 gimbal/yaw 标定。",
     "min": 0, "max": 50, "step": 0.5, "group": "plan"},
    {"key": "plan.yaw_offset_deg", "yaml_key": "yaw_offset_deg",
     "label": "yaw_offset", "name": "Yaw 固定零偏", "unit": "deg",
     "description": "补偿枪管、相机和云台之间的水平静态偏差。请先用静止靶标定，正负方向以画面实际偏差为准。",
     "min": -10, "max": 10, "step": 0.1, "group": "plan"},
    {"key": "plan.pitch_offset_deg", "yaml_key": "pitch_offset_deg",
     "label": "pitch_offset", "name": "Pitch 固定零偏", "unit": "deg",
     "description": "补偿俯仰方向的静态落点偏差，应在调空气阻力前完成标定，避免两个参数互相抵消。",
     "min": -10, "max": 10, "step": 0.1, "group": "plan"},
    {"key": "plan.fallback_bullet_speed", "yaml_key": "fallback_bullet_speed_mps",
     "label": "fallback_bullet_speed", "name": "回退弹速", "unit": "m/s",
     "description": "裁判系统弹速无效时用于继续计算跟随角；默认只跟随不允许开火。应接近当前弹种的实测平均弹速。",
     "min": 10, "max": 30, "step": 0.1, "group": "plan"},
    {"key": "target.q_translation", "yaml_key": "q_translation",
     "label": "q_translation", "name": "平移过程噪声", "unit": "",
     "description": "控制 EKF 对目标平移和高度变化的响应。调大更灵敏但轨迹更抖，调小更平滑但高速变向时容易滞后。",
     "min": 1, "max": 500, "step": 1, "group": "ekf"},
    {"key": "target.q_rotation", "yaml_key": "q_rotation",
     "label": "q_rotation", "name": "旋转过程噪声", "unit": "",
     "description": "控制 EKF 对整车 yaw 与角速度变化的响应。调大更容易跟上小陀螺，过大会放大姿态抖动。",
     "min": 1, "max": 2000, "step": 5, "group": "ekf"},
    {"key": "target.angle_variance", "yaml_key": "angle_variance",
     "label": "angle_variance", "name": "角度观测方差", "unit": "",
     "description": "方位角和俯仰角的观测噪声。调大表示更不信任 PnP 角度，曲线更稳但收敛更慢。",
     "min": 1e-4, "max": 1, "step": 1e-3, "group": "ekf"},
    {"key": "target.distance_variance_factor", "yaml_key": "distance_variance_factor",
     "label": "distance_var_factor", "name": "距离方差系数", "unit": "",
     "description": "距离观测方差随目标距离平方增长的系数。调大后远距离深度观测权重更低，能抑制 PnP 前后跳动。",
     "min": 0.001, "max": 1, "step": 0.005, "group": "ekf"},
    {"key": "target.armor_yaw_variance_base", "yaml_key": "armor_yaw_variance_base",
     "label": "yaw_var_base", "name": "装甲板 Yaw 方差", "unit": "",
     "description": "装甲板朝向观测的基础不确定度。调大可降低错误朝向对整车旋转状态的冲击，但会减慢朝向收敛。",
     "min": 0.001, "max": 1, "step": 0.005, "group": "ekf"},
    {"key": "fire.hit_margin_ratio", "yaml_key": "hit_margin_ratio",
     "label": "hit_margin_ratio", "name": "中心命中区域比例", "unit": "",
     "description": "只把装甲板中心这部分视为可命中。调小更保守，调大更容易放行开火；0.6 表示中央 60%。",
     "min": 0.1, "max": 1, "step": 0.05, "group": "fire"},
    {"key": "fire.min_yaw_tolerance_deg", "yaml_key": "min_yaw_tolerance_deg",
     "label": "min_yaw_tol", "name": "Yaw 最小容差", "unit": "deg",
     "description": "水平命中角容差的下限。调大更容易通过开火判定，但会降低边缘命中精度。",
     "min": 0, "max": 5, "step": 0.05, "group": "fire"},
    {"key": "fire.min_pitch_tolerance_deg", "yaml_key": "min_pitch_tolerance_deg",
     "label": "min_pitch_tol", "name": "Pitch 最小容差", "unit": "deg",
     "description": "垂直命中角容差的下限。调大更容易通过开火判定，过大会容忍明显的上下偏差。",
     "min": 0, "max": 5, "step": 0.05, "group": "fire"},
    {"key": "debug.overlay", "yaml_key": "overlay",
     "label": "overlay", "name": "本地叠加窗口", "unit": "", "kind": "bool",
     "description": "控制本机 OpenCV 调试窗口及叠加层。无显示器环境不要开启；部分运行方式需要重启后完全生效。",
     "group": "debug"},
    {"key": "debug.web_frame_every", "yaml_key": "web_frame_every",
     "label": "web_frame_every", "name": "网页画面抽帧", "unit": "帧",
     "description": "每隔 N 个相机帧编码一张网页 JPEG。数值越小越流畅、CPU 占用越高；实机建议从 5 开始。",
     "kind": "integer",
     "min": 1, "max": 30, "step": 1, "group": "debug"},
    {"key": "camera.exposure_ms", "yaml_key": "exposure_ms",
     "label": "exposure", "name": "相机曝光时间", "unit": "ms",
     "description": "曝光越长画面越亮、运动拖影越明显。需要与增益一起下发；该项仅修改当前运行，不写入 YAML。",
     "min": 0.1, "max": 20, "step": 0.1, "group": "camera", "persist": False},
    {"key": "camera.gain", "yaml_key": "gain",
     "label": "gain", "name": "相机模拟增益", "unit": "dB",
     "description": "增益越高画面越亮、噪点也越多。需要与曝光一起下发；该项仅修改当前运行，不写入 YAML。",
     "min": 0, "max": 20, "step": 0.1, "group": "camera", "persist": False},
]

FORCE_WORK_MODES = {"", "auto_aim", "outpost", "small_buff", "big_buff", "idle"}

# A copied module can supply the host application's tuning catalog as JSON.
_catalog_path = os.environ.get("VISION_TOOLS_PARAM_CATALOG")
if _catalog_path:
    with open(_catalog_path, "r", encoding="utf-8") as _catalog_file:
        _catalog = json.load(_catalog_file)
    if not isinstance(_catalog, list) or not all(
        isinstance(item, dict) and "key" in item and "yaml_key" in item
        for item in _catalog
    ):
        raise ValueError("VISION_TOOLS_PARAM_CATALOG must contain a parameter list")
    TUNABLE_PARAMS = _catalog
_force_modes = os.environ.get("VISION_TOOLS_FORCE_MODES")
if _force_modes is not None:
    FORCE_WORK_MODES = set(_force_modes.split(","))


def _read_yaml_scalar(text: str, yaml_key: str, kind: str = "number"):
    """按行读取未注释的 YAML 标量，保留原文件注释和排版。"""
    pattern = re.compile(rf"^[ \t]*{re.escape(yaml_key)}:[ \t]*(.+)$", re.M)
    match = pattern.search(text)
    if not match:
        return None
    raw = match.group(1).split("#", 1)[0].strip().strip('"').strip("'")
    if kind == "bool":
        lowered = raw.lower()
        if lowered in ("true", "yes", "on"):
            return True
        if lowered in ("false", "no", "off"):
            return False
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _read_yaml_text(text: str, yaml_key: str) -> str | None:
    pattern = re.compile(rf"^[ \t]*{re.escape(yaml_key)}:[ \t]*(.+)$", re.M)
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).split("#", 1)[0].strip().strip('"').strip("'")


def _nested_set(tree: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = tree
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def _nested_get(tree: dict, dotted: str):
    node = tree
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None, False
        node = node[key]
    return node, True


def _validate_payload(payload: dict, *, allow_force: bool = True,
                      persistent_only: bool = False) -> tuple[dict, dict]:
    by_key = {item["key"]: item for item in TUNABLE_PARAMS}
    accepted = {}
    rejected = {}
    for key, value in payload.items():
        if key == "debug.force_work_mode" and allow_force:
            if isinstance(value, str) and value in FORCE_WORK_MODES:
                accepted[key] = value
            else:
                rejected[key] = "必须是空串或已知 WorkMode"
            continue
        item = by_key.get(key)
        if not item:
            rejected[key] = "不在 Web 调参白名单"
            continue
        if persistent_only and item.get("persist", True) is False:
            rejected[key] = "仅支持运行时热更"
            continue
        kind = item.get("kind", "number")
        if kind == "bool":
            if not isinstance(value, bool):
                rejected[key] = "必须是布尔值"
                continue
            accepted[key] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            rejected[key] = "必须是数字"
            continue
        number = float(value)
        if not math.isfinite(number):
            rejected[key] = "必须是有限数字"
            continue
        if number < item["min"] or number > item["max"]:
            rejected[key] = f"超出范围 [{item['min']}, {item['max']}]"
            continue
        if kind == "integer":
            if not number.is_integer():
                rejected[key] = "必须是整数"
                continue
            accepted[key] = int(number)
        else:
            accepted[key] = number
    return accepted, rejected


def _read_command_flat() -> dict:
    try:
        command = read_json_cached(CMD_PATH)
    except Exception:  # noqa: BLE001
        return {}
    result = {}
    keys = [item["key"] for item in TUNABLE_PARAMS] + ["debug.force_work_mode"]
    for key in keys:
        value, exists = _nested_get(command, key)
        if exists:
            accepted, _ = _validate_payload({key: value})
            if key in accepted:
                result[key] = accepted[key]
    return result


def _command_tree(flat: dict) -> dict:
    tree = {}
    for key, value in flat.items():
        _nested_set(tree, key, value)
    return tree


def _write_json_atomic(path: str, value: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    with _json_lock:
        _json_cache.pop(path, None)


@app.route("/params")
def get_params():
    """滑条初值来自 YAML（启动真相），不是 cmd 文件。"""
    catalog = TUNABLE_PARAMS
    current = {}
    force_yaml = ""
    try:
        with open(YAML_PATH, "r", encoding="utf-8") as handle:
            text = handle.read()
        for item in catalog:
            key = item["key"]
            if item.get("persist", True) is False:
                value = None
            else:
                value = _read_yaml_scalar(text, item["yaml_key"], item.get("kind", "number"))
            current[key] = value
        force_yaml = _read_yaml_text(text, "force_work_mode") or ""
    except Exception:  # noqa: BLE001
        pass

    command = _read_command_flat()
    force = {"yaml": force_yaml}
    if "debug.force_work_mode" in command:
        force["cmd"] = command.pop("debug.force_work_mode")
    return no_store_json({"catalog": catalog, "yaml": current, "cmd": command,
                          "force_mode": force})


@app.route("/params", methods=["POST"])
def post_params():
    """校验并合并运行时覆盖，原子写入 cmd；绝不接受开火总闸门。"""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return no_store_json({"error": "expected JSON object"}, 400)

    accepted, rejected = _validate_payload(payload)
    if rejected:
        return no_store_json({"error": "参数校验失败", "rejected": rejected}, 400)
    if not accepted:
        return no_store_json({"error": "没有可写入的白名单参数"}, 400)

    merged = _read_command_flat()
    merged.update(accepted)
    tree = _command_tree(merged)

    try:
        _write_json_atomic(CMD_PATH, tree)
    except OSError as error:
        return no_store_json({"error": str(error)}, 500)
    return no_store_json({"ok": True, "accepted": list(accepted), "written": tree})


@app.route("/params/clear", methods=["POST"])
def post_params_clear():
    """清空 cmd 文件，runtime 停止继续应用上次热更（已应用的值不会自动回退）。"""
    try:
        _write_json_atomic(CMD_PATH, {})
    except OSError as error:
        return no_store_json({"error": str(error)}, 500)
    return no_store_json({"ok": True})


@app.route("/params/save", methods=["POST"])
def post_params_save():
    """把当前滑条值写回 config/auto_aim.yaml。与热更分开，避免误持久化。

    **按行外科替换**，不用 yaml.safe_dump：那份配置里每一段都有大段注释，
    整树重写会把注释洗光，下次上车就查不出参数含义。
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return no_store_json({"error": "expected JSON object"}, 400)

    accepted, rejected = _validate_payload(
        payload, allow_force=False, persistent_only=True)
    if rejected:
        return no_store_json({"error": "参数校验失败", "rejected": rejected}, 400)
    if not accepted:
        return no_store_json({"error": "没有可持久化的白名单参数"}, 400)

    try:
        with open(YAML_PATH, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        return no_store_json({"error": str(error)}, 500)

    by_key = {item["key"]: item for item in TUNABLE_PARAMS}
    written = []
    for key, value in accepted.items():
        item = by_key.get(key)
        if not item:
            continue
        yaml_key = item["yaml_key"]
        rendered = ("true" if value else "false") if item.get("kind") == "bool" else str(value)

        # 未注释：  key: 1.23
        pattern = re.compile(rf"^([ \t]*{re.escape(yaml_key)}:[ \t]*)(.+)$", re.M)
        new_text, count = pattern.subn(rf"\g<1>{rendered}", text, count=1)
        if count:
            text = new_text
            written.append(key)
            continue

        # 已注释：  # send_to_control_ms: 5.0 → 取消注释并写值
        commented = re.compile(
            rf"^([ \t]*)#[ \t]*{re.escape(yaml_key)}:[ \t]*.+$", re.M)
        new_text, count = commented.subn(rf"\g<1>{yaml_key}: {rendered}", text, count=1)
        if count:
            text = new_text
            written.append(key)

    missing = sorted(set(accepted) - set(written))
    if missing:
        return no_store_json({
            "error": "YAML 中找不到目标键，未写入任何内容",
            "missing": missing,
        }, 409)

    try:
        tmp = YAML_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, YAML_PATH)
    except OSError as error:
        return no_store_json({"error": str(error)}, 500)
    return no_store_json({"ok": True, "written": written})


# ---- 录制回放：series.jsonl + frames/ ----

RECORD_ROOT = os.environ.get(
    "VISION_TOOLS_RECORD_ROOT", os.environ.get("NEWVISION_RECORD_ROOT", "records")
)


@app.route("/replays")
def list_replays():
    """列出 RECORD_ROOT 下可回放的会话目录。"""
    sessions = []
    root = RECORD_ROOT
    try:
        for name in sorted(os.listdir(root), reverse=True):
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            series_path = os.path.join(path, "series.jsonl")
            frames_dir = os.path.join(path, "frames")
            n_frames = 0
            if os.path.isdir(frames_dir):
                n_frames = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
            n_samples = 0
            if os.path.isfile(series_path):
                with open(series_path, "rb") as handle:
                    n_samples = sum(1 for _ in handle)
            sessions.append({
                "name": name,
                "samples": n_samples,
                "frames": n_frames,
                "has_series": os.path.isfile(series_path),
            })
    except OSError:
        pass
    return no_store_json({"root": root, "sessions": sessions})


@app.route("/replays/<name>/series")
def replay_series(name):
    safe = os.path.basename(name)
    path = os.path.join(RECORD_ROOT, safe, "series.jsonl")
    if not os.path.isfile(path):
        return no_store_json({"error": "series not found"}, 404)
    points = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    points.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
    except OSError as error:
        return no_store_json({"error": str(error)}, 500)

    raw_count = len(points)
    max_points = clamp_int(request.args.get("max_points"), 4000, 100, 20000)
    if raw_count > max_points:
        # 画图只需均匀覆盖全程；原始 jsonl 保持不变，首尾样本必保留。
        indices = {
            round(index * (raw_count - 1) / (max_points - 1))
            for index in range(max_points)
        }
        points = [points[index] for index in sorted(indices)]

    # 拍平，与 /data 同构，前端 chart 逻辑可复用。
    keys = set()
    for row in points:
        stack = [("", row)]
        while stack:
            prefix, node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    key = f"{prefix}/{k}" if prefix else k
                    if isinstance(v, dict):
                        stack.append((key, v))
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        keys.add(key)
    keys.discard("t")
    series = {"time": []}
    for key in sorted(keys):
        series[key] = []
    for row in points:
        flat = {}

        def walk(node, prefix=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    key = f"{prefix}/{k}" if prefix else k
                    walk(v, key)
            elif isinstance(node, (int, float)) and not isinstance(node, bool):
                flat[prefix] = float(node)

        walk(row)
        series["time"].append(flat.get("t", len(series["time"])))
        for key in keys:
            series[key].append(flat.get(key, 0.0))
    return no_store_json({
        "name": safe,
        "series": series,
        "raw_count": raw_count,
        "returned_count": len(points),
        "sampled": len(points) != raw_count,
    })


@app.route("/replays/<name>/frame-times")
def replay_frame_times(name):
    """Return timestamps of encoded frames; gaps preserve dropped-frame time."""
    safe = os.path.basename(name)
    path = os.path.join(RECORD_ROOT, safe, "frames", "index.jsonl")
    frames = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    index = int(row["index"])
                    timestamp = float(row["t"])
                    if index >= 0 and math.isfinite(timestamp):
                        frames.append({"index": index, "t": timestamp})
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue
    except FileNotFoundError:
        # Older recordings have no frame index; the frontend uses legacy pacing.
        return no_store_json({"frames": [], "legacy": True})
    except OSError as error:
        return no_store_json({"error": str(error)}, 500)
    return no_store_json({"frames": frames, "legacy": False})


@app.route("/replays/<name>/frames/<int:index>")
def replay_frame(name, index):
    safe = os.path.basename(name)
    path = os.path.join(RECORD_ROOT, safe, "frames", f"frame_{index:06d}.jpg")
    if not os.path.isfile(path):
        return no_store_json({"error": "frame not found"}, 404)
    with open(path, "rb") as handle:
        data = handle.read()
    response = Response(data, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


def start_logger_relay() -> bool:
    """接收 aw_logger MessagePack，并向浏览器转发 JSON。

    WebsocketAppender 本身是客户端；这里仅补齐它已有协议的服务端角色，
    不重新采集、格式化或定义日志字段。依赖缺失时 Web 页面其余功能照常工作。
    """
    try:
        import asyncio
        import msgpack
        import websockets
    except ImportError as error:
        print(f"logger websocket relay disabled: {error}; run pip install -r requirements.txt")
        return False

    def run_relay() -> None:
        async def serve_forever() -> None:
            clients = set()
            browsers = set()

            async def send_json(peer, payload) -> bool:
                try:
                    await peer.send(json.dumps(payload, ensure_ascii=False))
                    return True
                except Exception:  # noqa: BLE001 — 断线由连接 finally 清理
                    return False

            async def handler(websocket) -> None:
                clients.add(websocket)
                try:
                    async for message in websocket:
                        if isinstance(message, bytes):
                            try:
                                entry = msgpack.unpackb(message, raw=False, strict_map_key=False)
                            except Exception as error:  # noqa: BLE001
                                print(f"logger relay ignored invalid MessagePack: {error}")
                                continue
                            if not isinstance(entry, dict):
                                continue
                            _logger_recent.append(entry)
                            dead = []
                            for peer in tuple(browsers):
                                if not await send_json(peer, {"type": "log", "entry": entry}):
                                    dead.append(peer)
                            for peer in dead:
                                browsers.discard(peer)
                                clients.discard(peer)
                            continue

                        try:
                            command = json.loads(message)
                        except (TypeError, json.JSONDecodeError):
                            continue
                        if command.get("client") in {"newvision-web", "fsvision-web"}:
                            browsers.add(websocket)
                            for entry in tuple(_logger_recent):
                                if not await send_json(websocket, {"type": "log", "entry": entry}):
                                    break
                            continue
                        if command.get("command") == "SET_LEVEL":
                            for peer in tuple(clients - browsers - {websocket}):
                                try:
                                    await peer.send(message)
                                except Exception:  # noqa: BLE001
                                    pass
                finally:
                    browsers.discard(websocket)
                    clients.discard(websocket)

            async with websockets.serve(
                handler,
                "0.0.0.0",
                LOGGER_WS_PORT,
                max_size=1024 * 1024,
                compression=None,
            ):
                print(f"aw_logger websocket relay: ws://0.0.0.0:{LOGGER_WS_PORT}")
                await asyncio.Future()

        try:
            asyncio.run(serve_forever())
        except OSError as error:
            print(f"logger websocket relay unavailable: {error}")

    threading.Thread(target=run_relay, name="aw-logger-relay", daemon=True).start()
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    init_shared_memory()
    threading.Thread(target=frame_reader_loop, daemon=True).start()
    start_logger_relay()

    url = f"http://{local_ip()}:{PORT}"
    print(f"fsvision web debugger: {url}")
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
