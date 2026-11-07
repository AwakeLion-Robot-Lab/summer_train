#!/usr/bin/env python3
"""Independently check saved images, YOLO labels, instance masks, crops and split separation."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root):
    root = Path(root).resolve()
    summary = json.loads((root / "summary.json").read_text())
    class_table = json.loads((root / "classes.json").read_text())["detection"]
    refs = json.loads((root / "reference" / "templates.json").read_text())
    references = {item["id"]: item for item in refs}
    seen_paths, hashes, objects_by_crop = set(), {}, {}
    image_counts, object_counts, negative_counts = Counter(), Counter(), Counter()
    class_counts = {split: Counter() for split in ("train", "val", "test")}
    min_color_contrast = 255
    for line in (root / "manifest.jsonl").read_text().splitlines():
        record = json.loads(line)
        image_path = root / record["image"]
        require(record["image"] not in seen_paths, f"Duplicate manifest entry: {image_path}")
        seen_paths.add(record["image"])
        split = record["split"]
        require(image_path.parent.name == split, f"Wrong split path: {image_path}")
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        require(digest not in hashes, f"Duplicate image bytes: {image_path}, {hashes.get(digest)}")
        hashes[digest] = record["image"]
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(root / record["mask"]), cv2.IMREAD_UNCHANGED)
        require(image is not None and mask is not None, f"Unreadable image or mask: {image_path}")
        require(mask.ndim == 2 and mask.dtype == np.uint8, f"Invalid instance mask: {image_path}")
        h, w = image.shape[:2]
        require([w, h] == record["size"] and mask.shape == (h, w), f"Shape mismatch: {image_path}")
        label_path = root / "labels" / split / (image_path.stem + ".txt")
        labels = label_path.read_text().splitlines()
        require(len(labels) == len(record["objects"]), f"Object count mismatch: {image_path}")
        ids = set(int(v) for v in np.unique(mask)) - {0}
        require(ids == set(range(1, len(labels)+1)), f"Instance IDs mismatch: {image_path}")
        for obj, label in zip(record["objects"], labels):
            values = label.split()
            require(len(values) == 5, f"Invalid YOLO row: {label_path}")
            cls = int(values[0])
            coordinates = np.array(values[1:], float)
            require(np.isfinite(coordinates).all(), f"Nonfinite box: {label_path}")
            cx, cy, bw, bh = coordinates
            require(0 < bw <= 1 and 0 < bh <= 1 and 0 <= cx-bw/2 and cx+bw/2 <= 1
                    and 0 <= cy-bh/2 and cy+bh/2 <= 1, f"Out-of-bounds box: {label_path}")
            decoded = np.array([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h])
            ys, xs = np.nonzero(mask == obj["instance_id"])
            require(len(xs) >= 12, f"Empty/tiny object mask: {image_path}")
            actual = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
            require(actual == obj["bbox_xyxy"] and np.max(np.abs(decoded-actual)) < 0.01,
                    f"Box does not match rendered support: {image_path}")
            require(cls == obj["class_id"] and class_table[str(cls)] == obj["class_name"],
                    f"Class mismatch: {image_path}")
            ref = references[obj["source_template"]]
            require(ref["color"] == obj["color"] and ref["digit"] == obj["digit"]
                    and ref["class_id"] == cls, f"Source provenance mismatch: {image_path}")
            target, opposite = (2, 0) if obj["color"] == "red" else (0, 2)
            pixels = image[mask == obj["instance_id"]].astype(float)
            contrast = float(np.percentile(pixels[:, target] - pixels[:, opposite], 90))
            min_color_contrast = min(min_color_contrast, contrast)
            require(contrast > 12, f"Glyph too weak or wrong color: {image_path}, {obj['class_name']}")
            crop = obj["classification_crop"]
            require(crop not in objects_by_crop, f"Repeated classification crop: {crop}")
            objects_by_crop[crop] = dict(image=record["image"], obj=obj)
            class_counts[split][obj["class_name"]] += 1
            object_counts[split] += 1
        if not labels:
            require(not mask.any(), f"Negative image has an object mask: {image_path}")
            negative_counts[split] += 1
        image_counts[split] += 1
    crop_counts, positive_crop_paths, all_crops = {}, set(), set()
    for line in (root / "classification_manifest.jsonl").read_text().splitlines():
        item = json.loads(line)
        path = root / item["path"]
        require(item["path"] not in all_crops, f"Repeated crop entry: {path}")
        all_crops.add(item["path"])
        image = cv2.imread(str(path))
        require(image is not None and image.shape == (64, 64, 3), f"Invalid classifier crop: {path}")
        require(path.parent.name == item["label"] and path.parent.parent.name == item["split"],
                f"Wrong classifier folder: {path}")
        require(item["parent_image"] in seen_paths, f"Missing crop parent: {path}")
        require(Path(item["parent_image"]).parent.name == item["split"], f"Crop crosses splits: {path}")
        if item["label"] != "negative":
            parent = objects_by_crop[item["path"]]
            obj = parent["obj"]
            require(parent["image"] == item["parent_image"] and str(obj["digit"]) == item["label"]
                    and obj["source_template"] == item["source_template"]
                    and obj["color"] == item["color"] and obj["instance_id"] == item["instance_id"],
                    f"Wrong crop label: {path}")
            positive_crop_paths.add(item["path"])
        crop_counts.setdefault(item["split"], Counter())[item["label"]] += 1
    require(positive_crop_paths == set(objects_by_crop), "Missing positive classification crops")
    for split, expected in summary["counts"].items():
        require(image_counts[split] == expected["images"] and object_counts[split] == expected["objects"]
                and negative_counts[split] == expected["negative_images"], f"Wrong totals: {split}")
        require(dict(class_counts[split]) == expected["by_class"], f"Wrong class totals: {split}")
        require(set(class_counts[split]) == set(class_table.values()), f"Missing classes: {split}")
        require(dict(crop_counts[split]) == summary["classification_counts"][split], f"Wrong crop totals: {split}")
        require(len(list((root / "images" / split).glob("*.jpg"))) == image_counts[split], f"Extra images: {split}")
        require(len(list((root / "labels" / split).glob("*.txt"))) == image_counts[split], f"Extra labels: {split}")
        require(len(list((root / "masks" / split).glob("*.png"))) == image_counts[split], f"Extra masks: {split}")
    report = dict(status="passed", detection_images=sum(image_counts.values()),
                  labeled_objects=sum(object_counts.values()), classification_crops=len(all_crops),
                  images_by_split=dict(image_counts), negative_images_by_split=dict(negative_counts),
                  minimum_p90_target_minus_opponent_channel=min_color_contrast,
                  identical_image_bytes=0,
                  limitation="Templates come from one source video shared by all splits; this is not a real-world evaluation.")
    (root / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    validate(parser.parse_args().dataset)
