#!/usr/bin/env python3
"""Validate simulated digit masks, panel projections, YOLO pose labels and PnP consistency."""

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


def decode_box(values, width, height):
    cx, cy, w, h = np.array(values, float)
    require(np.isfinite([cx, cy, w, h]).all() and w > 0 and h > 0, "Invalid normalized box")
    box = np.array([(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height])
    require(box.min() >= 0 and box[2] <= width and box[3] <= height, "Out-of-image box")
    return box


def mask_bounds(mask):
    y, x = np.nonzero(mask)
    require(len(x) >= 1, "Empty mask")
    return np.array([x.min(), y.min(), x.max()+1, y.max()+1])


def validate(root):
    cv2.setNumThreads(1)
    root = Path(root).resolve()
    summary = json.loads((root / "summary.json").read_text())
    config = json.loads((root / "assumptions.json").read_text())
    require(summary["geometry_status"] == config["geometry_status"] == "hypothesis_not_measured",
            "Missing hypothetical geometry disclosure")
    require(summary["no_recorded_rgb_patches_pasted"], "Unexpected renderer provenance")
    image_counts, object_counts, negatives = Counter(), Counter(), Counter()
    classes = {split: Counter() for split in ("train", "val", "test")}
    rows_used = {split: Counter() for split in ("train", "val", "test")}
    hashes, image_paths, crops = set(), {}, {}
    max_projection_error, max_pose_label_error, max_pnp_translation_error = 0.0, 0.0, 0.0
    minimum_color_contrast = 255.0
    for line in (root / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        split = row["split"]
        image_path = root / row["image"]
        require(row["image"] not in image_paths, f"Repeated image: {image_path}")
        image_paths[row["image"]] = split
        content = image_path.read_bytes()
        digest = hashlib.sha256(content).digest()
        require(digest not in hashes, f"Duplicate rendered image: {image_path}")
        hashes.add(digest)
        require(content == (root / "panel_pose" / row["image"]).read_bytes(), "Task image copies differ")
        image = cv2.imread(str(image_path))
        gm = cv2.imread(str(root / "masks" / split / (image_path.stem+".png")), cv2.IMREAD_UNCHANGED)
        pm = cv2.imread(str(root / "panel_pose" / "masks" / split / (image_path.stem+".png")), cv2.IMREAD_UNCHANGED)
        require(image is not None and gm is not None and pm is not None, "Unreadable image or mask")
        h, w = image.shape[:2]
        require([w, h] == row["image_size"] and gm.shape == pm.shape == (h, w), "Wrong image shape")
        digit_labels = (root / "labels" / split / (image_path.stem+".txt")).read_text().splitlines()
        pose_labels = (root / "panel_pose" / "labels" / split / (image_path.stem+".txt")).read_text().splitlines()
        objects = row["objects"]
        require(len(digit_labels) == len(pose_labels) == len(objects), "Task object count mismatch")
        expected_ids = set(range(len(objects)+1))
        require(set(np.unique(gm)) == set(np.unique(pm)) == expected_ids, "Wrong mask instance IDs")
        require(np.all(pm[gm > 0] == gm[gm > 0]), "A digit lies outside its simulated panel")
        camera = np.array(row["camera_matrix"], np.float64)
        require(camera.shape == (3, 3) and camera[0, 0] > 0 and camera[1, 1] > 0, "Invalid camera")
        for obj, digit_line, pose_line in zip(objects, digit_labels, pose_labels):
            instance_id = obj["instance_id"]
            digit_values, pose_values = digit_line.split(), pose_line.split()
            require(len(digit_values) == 5 and len(pose_values) == 17, "Wrong YOLO label shape")
            cls = int(digit_values[0])
            expected_cls = (0 if obj["color"] == "red" else 7)+obj["digit"]-1
            require(cls == obj["class_id"] == expected_cls, "Wrong digit/color mapping")
            digit_box = decode_box(digit_values[1:], w, h)
            actual_glyph_box = mask_bounds(gm == instance_id)
            require(np.max(np.abs(digit_box-actual_glyph_box)) < 0.01 and
                    obj["digit_bbox_xyxy"] == actual_glyph_box.tolist(), "Digit box differs from mask")
            require(int(pose_values[0]) == 0, "Pose task must use a single panel class")
            panel_box = decode_box(pose_values[1:5], w, h)
            require(np.max(np.abs(panel_box-np.array(obj["panel_bbox_xyxy"]))) < 0.01, "Wrong pose box")
            require(np.max(np.abs(panel_box-mask_bounds(pm == instance_id))) < 2.1, "Panel raster differs from pose box")
            kpts = np.array(pose_values[5:], float).reshape(4, 3)
            require((kpts[:, 2] == 2).all(), "Unexpected corner visibility")
            kpts_xy = kpts[:, :2]*[w, h]
            corners = np.array(obj["panel_corners_xy"], np.float64)
            label_error = float(np.max(np.abs(kpts_xy-corners)))
            require(label_error < 0.01, "Wrong corner coordinates in pose labels")
            points = np.array(obj["object_corners_m"], np.float64)
            rvec, tvec = np.array(obj["rotation_vector"]), np.array(obj["translation_m"])
            projected = cv2.projectPoints(points, rvec, tvec, camera, np.zeros(5))[0].reshape(4, 2)
            error = float(np.max(np.abs(projected-corners)))
            require(error < 1e-6, "3D projection disagrees with stored corners")
            success, recovered_rvec, recovered_tvec = cv2.solvePnP(
                points, corners, camera, np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
            require(success, "PnP failed for simulated plane")
            translation_error = float(np.linalg.norm(recovered_tvec.reshape(3)-tvec))
            require(translation_error < 1e-5, "PnP cannot recover simulated translation")
            require(obj["panel_size_m"] == [config["panel_width_m"], config["panel_height_m"]]
                    and obj["geometry_status"] == "hypothesis_not_measured", "Geometry metadata mismatch")
            require(obj.get("panel_type", "unspecified") == config.get("panel_type", "unspecified"),
                    "Panel type differs from selected geometry")
            target, opposite = (2, 0) if obj["color"] == "red" else (0, 2)
            pixels = image[gm == instance_id].astype(float)
            contrast = float(np.percentile(pixels[:, target]-pixels[:, opposite], 90))
            require(contrast > 15, f"Unreadably weak digit: {image_path}")
            minimum_color_contrast = min(minimum_color_contrast, contrast)
            crop = obj["classification_crop"]
            require(crop not in crops, "Repeated classification crop")
            crops[crop] = (row["image"], obj)
            led_rows = obj["appearance"]["led_rows"]
            require(led_rows in config["led_rows"][split], "Unexpected dot density for split")
            rows_used[split][str(led_rows)] += 1
            classes[split][obj["class_name"]] += 1
            object_counts[split] += 1
            max_projection_error = max(max_projection_error, error)
            max_pose_label_error = max(max_pose_label_error, label_error)
            max_pnp_translation_error = max(max_pnp_translation_error, translation_error)
        if not objects:
            negatives[split] += 1
        image_counts[split] += 1
    crop_counts, all_crops, positive_crops = {}, set(), set()
    for line in (root / "classification_manifest.jsonl").read_text().splitlines():
        item = json.loads(line)
        path = root / item["path"]
        require(item["path"] not in all_crops, "Repeated classification manifest entry")
        all_crops.add(item["path"])
        image = cv2.imread(str(path))
        require(image is not None and image.shape == (64, 64, 3), "Wrong classifier image shape")
        require(path.parent.name == item["label"] and path.parent.parent.name == item["split"], "Wrong crop folder")
        require(image_paths[item["parent_image"]] == item["split"], "Classifier crop crosses splits")
        if item["label"] != "negative":
            parent_image, obj = crops[item["path"]]
            require(parent_image == item["parent_image"] and str(obj["digit"]) == item["label"]
                    and obj["color"] == item["color"] and obj["instance_id"] == item["instance_id"], "Wrong crop provenance")
            positive_crops.add(item["path"])
        crop_counts.setdefault(item["split"], Counter())[item["label"]] += 1
    require(positive_crops == set(crops), "Missing positive crops")
    for split, totals in summary["counts"].items():
        require(image_counts[split] == totals["images"] and object_counts[split] == totals["objects"]
                and negatives[split] == totals["negative_images"], f"Wrong counts in {split}")
        require(dict(classes[split]) == totals["by_class"] and len(classes[split]) == 14, "Wrong class coverage")
        require(dict(rows_used[split]) == totals["led_rows"], "Wrong LED-density totals")
        require(dict(crop_counts[split]) == totals["classification_counts"], "Wrong classifier totals")
        for folder, suffix in (("images", "jpg"), ("labels", "txt"), ("masks", "png"),
                               ("panel_pose/images", "jpg"), ("panel_pose/labels", "txt"), ("panel_pose/masks", "png")):
            require(len(list((root / folder / split).glob("*."+suffix))) == image_counts[split], "Unexpected extra files")
    report = dict(status="passed", images=sum(image_counts.values()), simulated_panels=sum(object_counts.values()),
        classification_crops=len(all_crops), images_by_split=dict(image_counts),
        max_projection_error_px=max_projection_error, max_pose_label_rounding_error_px=max_pose_label_error,
        max_pnp_translation_error_m=max_pnp_translation_error, minimum_color_contrast=minimum_color_contrast,
        identical_image_bytes=0, geometry_status="hypothesis_not_measured",
        panel_type=config.get("panel_type", "unspecified"),
        panel_size_m=[config["panel_width_m"], config["panel_height_m"]], real_world_accuracy="not_evaluated")
    (root / "validation_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    validate(parser.parse_args().dataset)
