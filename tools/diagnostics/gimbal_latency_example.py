#!/usr/bin/env python3
"""固定 yaw 起点、不同步长的延迟采集示例。默认预览；加 --arm 才执行。"""

import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import subprocess

# 在这里改实验参数。示例限位不是实机标定值，请按你的云台修改。
INITIAL_RAD = 0.0
OFFSETS_RAD = [0.25, 0.50, 0.75]
REPEAT = 1000  # 调试先改成 5，正式采集再改回 500
LIMITS_RAD = (-0.80, 0.80)
DURATION_MS = 500  # 命令后的观测窗口，不是通信周期
ONSET_THRESHOLD_RAD = 0.005  # 约 0.286°；用静止噪声确认后再选定
SETTLE_TIMEOUT_MS = 5000
PORT = "/dev/ttyACM0"
BAUD = 115200


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--output", default="/tmp/yaw_latency_example")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    prefix = Path(args.output).expanduser().resolve()
    degrees = lambda value: format(math.degrees(value), ".12g")
    command = [
        "xmake", "run", "gimbal_step_csv", "--axis", "yaw",
        "--initials", degrees(INITIAL_RAD),
        "--offsets", ",".join(map(degrees, OFFSETS_RAD)),
        "--repeat", str(REPEAT),
        "--limits", ":".join(map(degrees, LIMITS_RAD)),
        "--duration-ms", str(DURATION_MS),
        "--onset-threshold-deg", degrees(ONSET_THRESHOLD_RAD),
        "--settle-timeout-ms", str(SETTLE_TIMEOUT_MS),
        "--port", PORT, "--baud", str(BAUD), "--output", str(prefix),
    ]
    if args.arm:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        # 保留阈值、窗口和实验参数，避免后续拟合混用不同定义的延迟。
        config = dict(initial_rad=INITIAL_RAD, offsets_rad=OFFSETS_RAD,
                      repeat=REPEAT, limits_rad=LIMITS_RAD, duration_ms=DURATION_MS,
                      onset_threshold_rad=ONSET_THRESHOLD_RAD,
                      settle_timeout_ms=SETTLE_TIMEOUT_MS, port=PORT, baud=BAUD)
        Path(str(prefix) + "_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8")
        command.append("--arm")
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)
    if not args.arm:
        return

    # 原工具的四份 CSV 保留；另外生成便于拟合的弧度版逐次结果。
    with Path(str(prefix) + "_trials.csv").open(newline="") as source, \
         Path(str(prefix) + "_latency.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["trial_id", "repeat", "initial_rad", "initial_actual_rad",
                         "step_rad", "target_rad", "onset_ms", "status",
                         "sample_count", "threshold_rad", "window_ms"])
        for row in csv.DictReader(source):
            status = ("observed" if row["onset_ms"] else
                      "no_feedback" if int(row["sample_count"]) == 0 else "not_observed")
            writer.writerow([row["trial_id"], row["repeat"],
                             *[math.radians(float(row[key])) for key in
                               ("initial_cmd_deg", "initial_actual_deg", "delta_cmd_deg", "target_deg")],
                             row["onset_ms"], status, row["sample_count"],
                             ONSET_THRESHOLD_RAD, DURATION_MS])
    print(f"用于拟合：{prefix}_latency.csv")


if __name__ == "__main__":
    main()
