#!/usr/bin/env python3
"""Passively sample an existing auto_aim process and its local UDP telemetry.

Never opens the camera/serial port, sends packets, or interrupts the process.
Refuses to share a UDP port that already has a listener.
"""
import argparse
import collections
import json
import math
import os
from pathlib import Path
import socket
import statistics
import time


def ticks(path):
    fields = path.read_text().rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])


def snapshot(proc):
    result = {}
    for task in (proc / "task").iterdir():
        try:
            result[task.name] = (ticks(task / "stat"), (task / "comm").read_text().strip())
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, help="Defaults to the sole running auto_aim process")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--port", type=int, default=9870)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if args.pid is None:
        candidates = []
        for path in Path("/proc").glob("[0-9]*/comm"):
            try:
                if path.read_text().strip() == "auto_aim":
                    candidates.append(int(path.parent.name))
            except (OSError, ProcessLookupError):
                pass
        if len(candidates) != 1:
            parser.error(f"Expected one auto_aim process; found {candidates}; use --pid")
        args.pid = candidates[0]
    proc = Path(f"/proc/{args.pid}")
    if (proc / "comm").read_text().strip() != "auto_aim":
        raise SystemExit("The selected process is not auto_aim")
    result = {"pid": args.pid, "cwd": str((proc / "cwd").resolve()),
              "executable": str((proc / "exe").resolve()),
              "cpu_count": os.cpu_count(), "wall_time": time.strftime("%F %T %z")}
    channel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    channel.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    try:
        channel.bind(("127.0.0.1", args.port))
        channel.settimeout(.1)
        result["udp"] = "exclusive listener; existing listeners are never displaced"
    except OSError as error:
        result["udp"] = str(error)
        channel.close()
        channel = None
    before = snapshot(proc)
    process_before = ticks(proc / "stat")
    start = time.monotonic()
    end = start + args.seconds
    values = collections.defaultdict(list)
    arrival = []
    frame_times = []
    modes = collections.Counter()
    packets = 0
    runtime_reports = []
    while time.monotonic() < end:
        if channel is None:
            time.sleep(min(.1, max(0, end - time.monotonic())))
            continue
        try:
            payload, _ = channel.recvfrom(65535)
            data = json.loads(payload)
        except socket.timeout:
            continue
        except (ValueError, UnicodeDecodeError):
            continue
        packets += 1
        report = data.get("runtime_perf")
        if isinstance(report, dict):
            runtime_reports.append(report)
        pipeline = data.get("pipeline")
        if not isinstance(pipeline, dict):
            continue
        arrival.append(time.monotonic())
        if isinstance(data.get("t"), (int, float)):
            frame_times.append(data["t"])
        modes[str(data.get("mode", "unknown"))] += 1
        for name, value in pipeline.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                values[name].append(float(value))
    elapsed = time.monotonic() - start
    hz = os.sysconf("SC_CLK_TCK")
    try:
        after = snapshot(proc)
        result["cpu_percent"] = (ticks(proc / "stat") - process_before) / hz / elapsed * 100
        result["threads"] = sorted([
            {"tid": tid, "name": name, "cpu_percent": (count - before[tid][0]) / hz / elapsed * 100}
            for tid, (count, name) in after.items() if tid in before
        ], key=lambda row: row["cpu_percent"], reverse=True)
        result["thread_count"] = len(after)
    except (FileNotFoundError, ProcessLookupError):
        result["process_exited"] = True
    if channel is not None:
        channel.close()
    result.update(seconds=elapsed, udp_packets=packets, pipeline_packets=len(arrival), modes=dict(modes))
    if len(arrival) > 1:
        # Pipeline messages may contain only rejected frames in detection-only
        # mode. Their packet rate is not the displayed/processed frame rate.
        result["pipeline_packet_hz"] = (len(arrival) - 1) / (arrival[-1] - arrival[0])
    if len(frame_times) > 1 and frame_times[-1] > frame_times[0]:
        result["frame_timestamp_fps"] = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
    result["pipeline"] = {}
    for name, samples in values.items():
        samples.sort()
        result["pipeline"][name] = {"mean": statistics.fmean(samples),
            "p50": statistics.median(samples), "p95": samples[min(len(samples)-1, int(len(samples)*.95))]}
    result["runtime_perf_windows"] = runtime_reports
    if runtime_reports:
        count = sum(row["frames"] for row in runtime_reports)
        result["runtime_summary"] = {
            "frames": count,
            "fps": count / sum(row["seconds"] for row in runtime_reports),
            "feedback_frames": sum(row["feedback_frames"] for row in runtime_reports)}
        for name, entry in runtime_reports[0].items():
            if isinstance(entry, dict) and "mean" in entry:
                result["runtime_summary"][name] = sum(
                    row[name]["mean"] * row["frames"] for row in runtime_reports) / count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    display = dict(result)
    display["threads"] = result.get("threads", [])[:8]
    display["runtime_perf_windows"] = len(runtime_reports)
    print(json.dumps(display, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
