#!/usr/bin/env python3
"""Export disjoint old-sticker recordings using the existing C++ detector/light refiner as teacher."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from depth_data import FEATURE_NAMES, geometry_features, glyph_image
from generate import contact_sheet, json_file, sha256_file


def validate(root):
    root = Path(root)
    rows = [json.loads(line) for line in (root/"manifest.jsonl").read_text().splitlines()]
    counts, digits, sequences, hashes = Counter(), Counter(), {}, {}
    ranges = {split: [] for split in ("train", "val", "test")}
    for row in rows:
        split, sequence = row["split"], row["sequence"]
        if sequence in sequences and sequences[sequence] != split:
            raise ValueError(f"Recording crosses splits: {sequence}")
        sequences[sequence] = split
        crop = cv2.imread(str(root/row["crop"]))
        x0, y0, x1, y1 = row["digit_bbox_xyxy"]
        if crop is None or crop.shape[:2] != (y1-y0, x1-x0):
            raise ValueError(f"Crop/coordinates mismatch: {row['id']}")
        if row["panel_type"] != "small" or row["label_source"] != "lightbar_refined_ippe_pseudo_label":
            raise ValueError("Unexpected source or large armor in old-sticker dataset")
        if not np.isfinite(row["xyz_camera_m"]).all() or row["depth_z_m"] <= 0:
            raise ValueError("Invalid teacher distance")
        if abs(row["depth_z_m"]-row["xyz_camera_m"][2]) > 1e-9:
            raise ValueError("Optical-axis depth confused with range")
        if abs(np.linalg.norm(row["xyz_camera_m"])-row["range_m"]) > 1e-9:
            raise ValueError("Wrong Euclidean range")
        if row["teacher_rmse_px"] > 1.5 or row["teacher_alternative_depth_relative_gap"] > .10:
            raise ValueError("Teacher quality gate not respected")
        k = np.asarray(row["camera_matrix"], float)
        width, height = row["panel_size_m"]
        object_points = np.array([[-width/2, -height/2, 0], [width/2, -height/2, 0],
                                  [width/2, height/2, 0], [-width/2, height/2, 0]], float)
        projected = cv2.projectPoints(object_points, np.asarray(row["teacher_rvec"]),
            np.asarray(row["xyz_camera_m"]), k, np.asarray(row["distortion_coefficients"]))[0].reshape(4, 2)
        rmse = np.sqrt(np.mean(np.sum((projected-np.asarray(row["teacher_corners_xy"]))**2, axis=1)))
        if abs(rmse-row["teacher_rmse_px"]) > 1e-4:
            raise ValueError("Teacher distance is not paired with its own image points")
        geometry_features(row["digit_bbox_xyxy"], row["camera_matrix"])
        glyph_image(crop)
        digest = hashlib.sha256((root/row["crop"]).read_bytes()).hexdigest()
        if digest in hashes and hashes[digest] != split:
            raise ValueError("Identical crop bytes cross data splits")
        hashes[digest] = split
        counts[split] += 1
        digits[str(row["digit"])] += 1
        ranges[split].append(row["depth_z_m"])
    if not all(counts[s] > 0 for s in ("train", "val", "test")):
        raise ValueError("No accepted samples in one or more recording-level splits")
    report = dict(status="passed", samples=len(rows), samples_by_split=dict(counts), digits=dict(digits),
                  recordings=sequences, depth_z_by_split={s: dict(min_m=min(z), max_m=max(z)) for s,z in ranges.items()},
                  label_status="PnP pseudo labels, not independent distance truth",
                  limitations=["same-scene recordings", "narrow distance coverage", "unverified recording-calibration correspondence"])
    json_file(root/"validation_report.json", report)
    return report


def export(args):
    config = json.loads(Path(args.recordings).read_text())
    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output must be empty")
    if args.stride <= 0 or args.max_frames < 0:
        raise ValueError("Invalid sampling parameters")
    root.mkdir(parents=True, exist_ok=True)
    all_rows, results = [], []
    sequence_ids = set()
    for spec in config["recordings"]:
        sequence = spec["sequence"]
        if sequence in sequence_ids:
            raise ValueError("Repeated sequence ID")
        sequence_ids.add(sequence)
        destination = root/"sequences"/sequence
        log_path = root/(sequence+".log")
        command = [str(Path(args.binary).resolve()), str(Path(spec["video"]).resolve()),
            f"--output={destination}", f"--split={spec['split']}", f"--sequence={sequence}",
            f"--config={Path(config['config']).resolve()}", f"--calibration={Path(config['calibration']).resolve()}",
            f"--model={Path(config['model']).resolve()}", f"--stride={args.stride}", f"--max-frames={args.max_frames}"]
        print("Exporting", sequence, spec["split"], flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
        if result.returncode:
            raise RuntimeError(f"Exporter failed for {sequence}; see {log_path}")
        summary = json.loads((destination/"summary.json").read_text())
        summary["source_video_sha256"] = sha256_file(spec["video"])
        results.append(summary)
        for line in (destination/"manifest.jsonl").read_text().splitlines():
            row = json.loads(line)
            row["crop"] = "sequences/"+sequence+"/"+row["crop"]
            all_rows.append(row)
        print(sequence, "samples", summary["samples"], "decoded", summary["decoded_frames"], flush=True)
    (root/"manifest.jsonl").write_text("".join(json.dumps(row)+"\n" for row in all_rows))
    model_path=Path(config["model"])
    model_files=[model_path]
    if model_path.suffix.lower()==".xml":
        model_files.append(model_path.with_suffix(".bin"))
    json_file(root/"summary.json", dict(source="old_printed_sticker_recordings_only", recordings=results,
        student_features=FEATURE_NAMES, stride=args.stride, max_frames_per_recording=args.max_frames,
        calibration_sha256=sha256_file(config["calibration"]), model_sha256=sha256_file(config["model"]),
        model_file_sha256={str(path):sha256_file(path) for path in model_files},
        exporter_sha256=sha256_file(args.binary), recording_config=config,
        student_inputs="raw glyph crop and digit-box-derived camera features; no teacher lightbar coordinates"))
    samples = []
    for split in ("train", "val", "test"):
        subset = [r for r in all_rows if r["split"] == split]
        for row in subset[::max(1, len(subset)//14)][:14]:
            image = cv2.imread(str(root/row["crop"]))
            samples.append((image, f"{split} {row['digit']} Z={row['depth_z_m']:.2f}m"))
    contact_sheet(samples, root/"preview.jpg", columns=7, tile_size=160)
    report = validate(root)
    generator = root/"generator"
    generator.mkdir()
    for name in ("export_sticker_depth.py", "depth_data.py", "train_sticker_depth.py",
                 "sticker_depth_model.py", "predict_sticker_depth.py", "export_depth.cpp", "depth_teacher.hpp", "STICKER_DEPTH.md"):
        shutil.copyfile(Path(__file__).with_name(name), generator/name)
    shutil.copyfile(args.recordings, generator/"sticker_recordings.json")
    shutil.copyfile(Path(__file__).with_name("STICKER_DEPTH.md"), root/"README.md")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", default=str(Path(__file__).with_name("sticker_recordings.json")))
    parser.add_argument("--binary", default="build/linux/x86_64/release/armor_depth_export")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=1800)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--validate-only", action="store_true")
    options = parser.parse_args()
    if options.validate_only:
        print(json.dumps(validate(options.output), indent=2))
    else:
        export(options)
