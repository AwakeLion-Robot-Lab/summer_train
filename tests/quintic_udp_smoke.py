#!/usr/bin/env python3
"""Verify the real probe executable's UDP JSON and CSV schema on loopback."""
import csv
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


def main():
    root = Path(__file__).resolve().parents[1]
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (
        root / "build/linux/x86_64/release/daedalus_quintic_probe")
    with tempfile.TemporaryDirectory(prefix="quintic-udp-") as tmp, socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(0.2)
        csv_path = Path(tmp) / "trace.csv"
        process = subprocess.Popen([
            str(binary), "--synthetic", "--realtime", "--duration=1",
            f"--port={receiver.getsockname()[1]}", f"--csv={csv_path}"],
            cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        packets = []
        deadline = time.monotonic() + 15
        try:
            while time.monotonic() < deadline:
                try:
                    payload, _ = receiver.recvfrom(65535)
                    packet = json.loads(payload)
                    assert len(payload) < 65507
                    assert math.isfinite(packet["t"])
                    assert packet["source"]["synthetic"] == 1
                    assert packet["source"]["command_sent"] == 0
                    assert "planned_velocity_rad_s" in packet["yaw"]
                    assert "planned_acceleration_rad_s2" in packet["pitch"]
                    assert "segment_acc_feasible" in packet["blend"]
                    assert "end" in packet["boundary"] and "end_valid" in packet["join"]
                    assert packet["yaw"]["measured_rad"] is None
                    if packets:
                        assert packet["t"] > packets[-1]["t"]
                    packets.append(packet)
                except socket.timeout:
                    if process.poll() is not None:
                        break
            output = process.communicate(timeout=3)[0]
            assert process.returncode == 0, output
            assert len(packets) >= 100, f"only {len(packets)} packets: {output}"
            rows = list(csv.DictReader(csv_path.open()))
            assert len(rows) == 200
            assert len(rows[0]) == len(set(rows[0]))
            by_frame = {int(row["/source/frame"]): row for row in rows}
            for packet in packets:
                row = by_frame[packet["source"]["frame"]]
                assert float(row["t"]) == packet["t"]
                assert float(row["/yaw/planned_rad"]) == packet["yaw"]["planned_rad"]
                assert row["/yaw/measured_rad"] == ""
            assert packets[-1]["counts"]["commits"] > 0
            print(f"UDP smoke passed: {len(packets)} JSON packets, {len(rows)} CSV rows")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
