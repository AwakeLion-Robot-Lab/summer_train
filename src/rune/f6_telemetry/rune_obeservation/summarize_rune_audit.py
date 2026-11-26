import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def distribution(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": ordered[-1],
    }


def summarize(path):
    counters = Counter()
    feature_counts = {stage: Counter() for stage in ("raw", "stable")}
    timings = {name: [] for name in ("detect_ms", "refine_ms", "total_ms", "replay_lag_ms")}
    previous_time = None
    backend = None
    replay_clock = None
    with path.open() as stream:
        for index, line in enumerate(stream):
            record = json.loads(line)
            if record["frame"] != index:
                raise ValueError("audit frames must be consecutive and start at zero")
            timestamp = record["t"]
            if not math.isfinite(timestamp) or (previous_time is not None and timestamp <= previous_time):
                raise ValueError("audit timestamps must increase")
            previous_time = timestamp
            if index == 0:
                backend, replay_clock = record["backend"], record["replay_clock"]
            if record["backend"] != backend or record["replay_clock"] != replay_clock:
                raise ValueError("audit backend and replay clock must remain consistent")
            counters["frames"] += 1
            counters["positive_frames"] += bool(record["raw"])
            counters["detections"] += len(record["raw"])
            for name in ("detect_ms", "refine_ms", "replay_lag_ms"):
                value = record.get(name)
                if value is not None:
                    if not math.isfinite(value):
                        raise ValueError(f"invalid timing in frame {index}: {name}")
                    timings[name].append(value)
            total = record["detect_ms"] + record["refine_ms"]
            timings["total_ms"].append(total)
            width, height = record["image_size"]
            for stage, counts in feature_counts.items():
                for feature in record[stage]:
                    prefix = f"class_{feature['class']}"
                    counts[f"{prefix}_detections"] += 1
                    points = [(f"gap_{i}", point) for i, point in enumerate(feature["gaps"])
                              if point is not None]
                    points.extend((f"fan_{identity}", point) for identity, point in feature["fan"].items())
                    for identity, point in points:
                        x, y = point
                        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
                            raise ValueError(f"invalid feature in frame {index}: {stage}/{identity}")
                        counts[f"{prefix}_{identity}"] += 1
    if not counters["frames"]:
        raise ValueError("audit contains no frames")
    return {
        "audit": str(path), "backend": backend, "replay_clock": replay_clock,
        "counts": dict(counters),
        "timings": {name: distribution(values) for name, values in timings.items()},
        "feature_availability": {stage: dict(counts) for stage, counts in feature_counts.items()},
        "interpretation": "特征计数表示可见测量的数量。误检、漏检、编号正确率与定位误差需要人工标注。",
    }


def main():
    parser = argparse.ArgumentParser(description="汇总能量机关离线观测和耗时")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(summarize(args.audit), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
    else:
        print(report)


if __name__ == "__main__":
    main()
