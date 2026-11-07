#!/usr/bin/env python3
"""Render new LED panels with a 3D planar camera model and explicitly hypothetical geometry."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np

from generate import (NAMES, SPLITS, bounds, classifier_crop, contact_sheet,
                      json_file, sha256_file, write_image)


def check_config(config):
    if config["geometry_status"] != "hypothesis_not_measured":
        raise ValueError("This renderer requires geometry to be marked hypothetical")
    for key in ("panel_width_m", "panel_height_m"):
        if not 0.01 < config[key] < 1.0:
            raise ValueError(f"Invalid {key}")
    for key in ("digit_height_ratio", "digit_width_scale", "led_radius_pitch_ratio",
                "yaw_deg", "pitch_deg", "roll_deg", "focal_length_over_image_width",
                "requested_panel_height_px"):
        value = config[key]
        if len(value) != 2 or not np.isfinite(value).all() or value[0] > value[1]:
            raise ValueError(f"Invalid range {key}")
    if not 0.2 <= config["digit_height_ratio"][0] <= config["digit_height_ratio"][1] <= 0.9:
        raise ValueError("Digit height ratio must fit inside the simulated panel")
    for split in SPLITS:
        if not config["led_rows"][split] or min(config["led_rows"][split]) < 16:
            raise ValueError(f"Invalid LED rows for {split}")


def reconstruct_glyphs(reference, output):
    records = json.loads((reference / "templates.json").read_text())
    glyphs, provenance, tiles = {}, {}, []
    for digit in range(1, 8):
        normalized, source_ids = [], []
        for record in records:
            if record["digit"] != digit:
                continue
            path = reference / "crops" / (record["id"] + ".png")
            crop = cv2.imread(str(path))
            if crop is None:
                raise ValueError(f"Cannot read source crop: {path}")
            # The saturated main-color halo is too broad to recover strokes. The green
            # channel isolates the yellow/cyan LED core in this particular recording.
            opposite = 0 if record["color"] == "red" else 2
            signal = np.maximum(crop[:, :, 1].astype(np.float32)-0.3*crop[:, :, opposite], 0)
            signal = cv2.GaussianBlur(signal, (0, 0), 0.5)
            if signal.max() < 30:
                raise ValueError(f"No clear luminous core: {path}")
            mask = (signal > 0.36*signal.max()).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
            x0, y0, x1, y1 = bounds(mask)
            width = max(5, round(144*(x1-x0)/(y1-y0)))
            if width > 160:
                raise ValueError(f"Implausibly wide source glyph: {record['id']}")
            scaled = cv2.resize(mask[y0:y1, x0:x1].astype(np.float32), (width, 144))
            canvas = np.zeros((176, 176), np.float32)
            x = (canvas.shape[1]-width)//2
            canvas[16:160, x:x+width] = scaled
            normalized.append(canvas)
            source_ids.append(dict(id=record["id"], crop_sha256=sha256_file(path)))
        if not normalized:
            raise ValueError(f"No reviewed source for digit {digit}")
        consensus = np.median(np.stack(normalized), axis=0)
        glyph = (cv2.GaussianBlur(consensus, (0, 0), 0.7) > 0.42).astype(np.uint8)
        glyphs[digit] = glyph
        provenance[str(digit)] = source_ids
        write_image(output / "reference" / "silhouettes" / f"{digit}.png", glyph*255)
        tiles.append((cv2.cvtColor(glyph*255, cv2.COLOR_GRAY2BGR), f"reconstructed {digit}"))
    contact_sheet(tiles, output / "reference" / "silhouettes.jpg", columns=7)
    json_file(output / "reference" / "silhouette_sources.json", provenance)
    return glyphs, provenance


def panel_texture(glyph, digit, color, rng, config, split):
    aspect = config["panel_width_m"] / config["panel_height_m"]
    th = 288
    tw = round(th*aspect)
    rows = int(rng.choice(config["led_rows"][split]))
    cols = round(rows*aspect)
    height_ratio = float(rng.uniform(*config["digit_height_ratio"]))
    width_scale = float(rng.uniform(*config["digit_width_scale"]))
    x0, y0, x1, y1 = bounds(glyph)
    gh = round(rows*height_ratio)
    gw = max(3, round(gh*(x1-x0)/(y1-y0)*width_scale))
    if gw >= cols-4:
        raise ValueError("Configured digit width does not fit panel")
    grid_glyph = cv2.resize(glyph[y0:y1, x0:x1].astype(np.float32), (gw, gh),
                            interpolation=cv2.INTER_AREA) > 0.40
    active = np.zeros((rows, cols), bool)
    jitter = config["digit_center_jitter_ratio"]
    dx, dy = rng.uniform(-jitter, jitter, 2)
    gx, gy = round((cols-gw)/2+dx*cols), round((rows-gh)/2+dy*rows)
    gx, gy = min(cols-gw-1, max(1, gx)), min(rows-gh-1, max(1, gy))
    active[gy:gy+gh, gx:gx+gw] = grid_glyph
    yy, xx = np.indices((th, tw), dtype=np.float32)
    grid_x, grid_y = xx/tw*cols, yy/th*rows
    cells_x = np.minimum(grid_x.astype(int), cols-1)
    cells_y = np.minimum(grid_y.astype(int), rows-1)
    offset_x, offset_y = np.abs(grid_x % 1-0.5), np.abs(grid_y % 1-0.5)
    radius = float(rng.uniform(*config["led_radius_pitch_ratio"]))
    shape = str(rng.choice(["round", "square"]))
    metric = np.sqrt(offset_x**2+offset_y**2) if shape == "round" else np.maximum(offset_x, offset_y)
    led = np.clip((radius-metric)*min(tw/cols, th/rows)+0.5, 0, 1)
    # New radiance is generated from cells; no recorded RGB pixels are used here.
    brightness = rng.uniform(0.85, 1.15, (rows, cols)).astype(np.float32)
    scan_period = 0
    if rng.random() < 0.25:
        scan_period = int(rng.integers(5, 13))
        brightness[int(rng.integers(scan_period))::scan_period] *= rng.uniform(0.45, 0.80)
    dropout = rng.random((rows, cols)) < rng.uniform(0, 0.01)
    brightness[dropout] *= 0.30
    density = led*active[cells_y, cells_x]*brightness[cells_y, cells_x]
    glyph_mask = (led > 0.25) & active[cells_y, cells_x]
    emitted = np.array([0.015, rng.uniform(0.10, 0.19), 1.0] if color == "red"
                       else [1.0, rng.uniform(0.18, 0.28), 0.015], np.float32)
    emission = density[:, :, None]*emitted[None, None, :]*rng.uniform(2.1, 4.2)
    # Screen reflectance and faint unlit cells are assumptions, not recovered hardware details.
    surface = rng.uniform(0.008, 0.045)
    texture = np.repeat(np.full((th, tw, 1), surface, np.float32), 3, axis=2)
    texture *= rng.uniform(0.8, 1.2, (1, 1, 3)).astype(np.float32)
    texture += led[:, :, None]*rng.uniform(0.001, 0.008)
    border_visible = bool(rng.random() < 0.55)
    if border_visible:
        width = int(rng.integers(1, 5))
        border_shade = float(surface*rng.uniform(1.5, 3.5))
        cv2.rectangle(texture, (0, 0), (tw-1, th-1), (border_shade,)*3, width)
    light_direction = float(rng.uniform(-0.35, 0.35))
    texture *= (1+light_direction*(xx/tw-0.5))[:, :, None]
    params = dict(digit=digit, color=color, led_rows=rows, led_columns=cols,
                  led_shape=shape, led_radius_pitch_ratio=radius, digit_height_ratio=height_ratio,
                  digit_width_scale=width_scale, digit_grid_xywh=[gx, gy, gw, gh],
                  approximate_scan_period_rows=scan_period, border_visible=border_visible)
    return texture, emission, glyph_mask.astype(np.uint8), params


def rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    return rz @ ry @ rx


def world_corners(config):
    w, h = config["panel_width_m"], config["panel_height_m"]
    return np.array([[-w/2, -h/2, 0], [w/2, -h/2, 0], [w/2, h/2, 0], [-w/2, h/2, 0]], np.float64)


def project(points, rotation, translation, camera):
    camera_points = points @ rotation.T + translation
    homogeneous = camera_points @ camera.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def arena_background(rng, size):
    yy, xx = np.indices((size, size), dtype=np.float32)
    horizon = float(rng.uniform(0.30, 0.50)*size)
    vanishing_x = float(rng.uniform(0.30, 0.70)*size)
    wall = rng.uniform(0.02, 0.12, 3).astype(np.float32)
    floor = rng.uniform(0.04, 0.20, 3).astype(np.float32)
    gradient = np.clip((yy-horizon)/(size-horizon), 0, 1)
    scene = np.empty((size, size, 3), np.float32)
    scene[:] = wall
    scene = scene*(1-gradient[:, :, None]) + floor*gradient[:, :, None]
    grid_color = tuple(float(v) for v in floor*rng.uniform(0.75, 1.25))
    for x in np.linspace(-size, size*2, 12):
        cv2.line(scene, (round(vanishing_x), round(horizon)), (round(x), size-1), grid_color, 1)
    for distance in (1.15, 1.4, 1.9, 2.8, 4.5, 8.0):
        y = round(horizon+(size-horizon)/distance)
        cv2.line(scene, (0, y), (size-1, y), grid_color, 1)
    # Procedural cuboids provide perspective clutter, not a claim about a particular arena.
    for _ in range(int(rng.integers(5, 12))):
        x = int(rng.integers(-40, size-20))
        y = int(rng.integers(round(horizon), size-5))
        width, height = (int(v) for v in rng.integers(20, max(21, size//4), 2))
        depth = int(rng.integers(6, max(7, width//3)))
        shade = rng.uniform(0.015, 0.25, 3)
        front = np.array([[x, y-height], [x+width, y-height], [x+width, y], [x, y]], np.int32)
        top = np.array([[x, y-height], [x+depth, y-height-depth],
                        [x+width+depth, y-height-depth], [x+width, y-height]], np.int32)
        side = np.array([[x+width, y-height], [x+width+depth, y-height-depth],
                         [x+width+depth, y-depth], [x+width, y]], np.int32)
        for poly, gain in ((front, 1), (top, 1.25), (side, 0.65)):
            cv2.fillConvexPoly(scene, poly, tuple(float(v) for v in shade*gain))
    for _ in range(int(rng.integers(3, 10))):
        center = tuple(int(v) for v in rng.integers(0, size, 2))
        radius = int(rng.integers(1, 5))
        cv2.circle(scene, center, radius, tuple(float(v) for v in rng.uniform(0.05, 0.8, 3)), -1)
    scene += cv2.resize(rng.normal(0, 0.008, (32, 32, 3)).astype(np.float32), (size, size))
    return np.maximum(cv2.GaussianBlur(scene, (0, 0), rng.uniform(0.4, 1.4)), 0)


def new_camera(rng, size, config):
    focal = float(rng.uniform(*config["focal_length_over_image_width"])*size)
    fy = focal*float(rng.uniform(0.98, 1.02))
    cx, cy = size/2+rng.uniform(-0.03, 0.03, 2)*size
    return np.array([[focal, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)


def render_panel(scene, glyph_mask, panel_mask, glyphs, config, split, rng, camera,
                 class_id, instance_id, reserved, fixed_pose=None):
    size = scene.shape[0]
    digit, color = class_id % 7+1, "red" if class_id < 7 else "blue"
    texture, emission, mask, params = panel_texture(glyphs[digit], digit, color, rng, config, split)
    th, tw = mask.shape
    local_corners = world_corners(config)
    for _ in range(80):
        if fixed_pose is None:
            yaw = float(rng.uniform(*config["yaw_deg"]))
            pitch = float(rng.uniform(*config["pitch_deg"]))
            roll = float(rng.uniform(*config["roll_deg"]))
            target_height = float(np.exp(rng.uniform(*np.log(config["requested_panel_height_px"]))))
            z = camera[1, 1]*config["panel_height_m"]/target_height
            u, v = rng.uniform(0.13, 0.87, 2)*size
            translation = np.array([(u-camera[0, 2])*z/camera[0, 0],
                                    (v-camera[1, 2])*z/camera[1, 1], z])
        else:
            yaw, pitch, roll, u, v, z = fixed_pose
            translation = np.array([(u-camera[0, 2])*z/camera[0, 0],
                                    (v-camera[1, 2])*z/camera[1, 1], z])
        rotation = rotation_matrix(yaw, pitch, roll)
        corners = project(local_corners, rotation, translation, camera)
        if (corners < 8).any() or (corners > size-9).any():
            if fixed_pose is not None:
                raise ValueError("Fixed preview pose falls outside image")
            continue
        x0, y0 = np.floor(corners.min(axis=0)-4).astype(int)
        x1, y1 = np.ceil(corners.max(axis=0)+5).astype(int)
        if reserved[y0:y1, x0:x1].any():
            continue
        # The radiance texture is projected using the actual sampled 3D plane.
        source = np.float32([[0, 0], [tw-1, 0], [tw-1, th-1], [0, th-1]])
        destination = np.float32(corners-[x0, y0])
        homography = cv2.getPerspectiveTransform(source, destination)
        width, height = x1-x0, y1-y0
        surface = cv2.warpPerspective(texture, homography, (width, height))
        light = cv2.warpPerspective(emission, homography, (width, height))
        warped_glyph = cv2.warpPerspective(mask, homography, (width, height), flags=cv2.INTER_NEAREST)
        warped_panel = cv2.warpPerspective(np.ones_like(mask), homography, (width, height),
                                         flags=cv2.INTER_NEAREST)
        if np.count_nonzero(warped_glyph) < 45:
            continue
        local_box = bounds(warped_glyph)
        if local_box[3]-local_box[1] < 14 or local_box[2]-local_box[0] < 7:
            continue
        roi = scene[y0:y1, x0:x1]
        roi[warped_panel > 0] = surface[warped_panel > 0]
        roi += light
        glyph_mask[y0:y1, x0:x1][warped_glyph > 0] = instance_id
        panel_mask[y0:y1, x0:x1][warped_panel > 0] = instance_id
        reserved[y0:y1, x0:x1] = 1
        box = [int(local_box[0]+x0), int(local_box[1]+y0), int(local_box[2]+x0), int(local_box[3]+y0)]
        rvec = cv2.Rodrigues(rotation)[0].reshape(3)
        return dict(instance_id=instance_id, class_id=class_id, class_name=NAMES[class_id],
                    digit=digit, color=color, digit_bbox_xyxy=box,
                    panel_type=config.get("panel_type", "unspecified"),
                    panel_bbox_xyxy=[float(corners[:, 0].min()), float(corners[:, 1].min()),
                                     float(corners[:, 0].max()), float(corners[:, 1].max())],
                    panel_corners_xy=corners.tolist(), corner_visibility=[2, 2, 2, 2],
                    panel_size_m=[config["panel_width_m"], config["panel_height_m"]],
                    object_corners_m=local_corners.tolist(), rotation_vector=rvec.tolist(),
                    translation_m=translation.tolist(), yaw_pitch_roll_deg=[yaw, pitch, roll],
                    geometry_status="hypothesis_not_measured", appearance=params)
    return None


def camera_image(scene, rng):
    size = scene.shape[0]
    exposure = float(rng.uniform(0.70, 1.35))
    scene = scene*exposure
    sigma = float(rng.uniform(0.25, 0.85))
    bloom = cv2.GaussianBlur(np.maximum(scene-0.4, 0), (0, 0), float(rng.uniform(1.2, 3.2)))
    scene = cv2.GaussianBlur(scene, (0, 0), sigma) + bloom*rng.uniform(0.15, 0.40)
    yy, xx = np.indices((size, size), dtype=np.float32)
    vignette = 1-rng.uniform(0.05, 0.25)*(((xx-size/2)/size)**2+((yy-size/2)/size)**2)
    scene *= vignette[:, :, None]
    shot_noise = rng.normal(0, 1, scene.shape).astype(np.float32)*np.sqrt(np.maximum(scene, 0))*0.004
    read_noise = rng.normal(0, 0.002, (size, size, 1)).astype(np.float32)
    gamma = float(rng.uniform(0.75, 1.05))
    image = (np.clip(scene+shot_noise+read_noise, 0, 1)**gamma*255).astype(np.uint8)
    return image, dict(exposure_gain=exposure, gaussian_psf_sigma_px=sigma, gamma=gamma,
                       jpeg_quality=int(rng.integers(82, 97)), distortion_coefficients=[0, 0, 0, 0, 0])


def make_scene(rng, size, glyphs, config, split, primary, negative):
    scene = arena_background(rng, size)
    camera = new_camera(rng, size, config)
    glyph_mask = np.zeros((size, size), np.uint8)
    panel_mask, reserved = glyph_mask.copy(), glyph_mask.copy()
    objects = []
    desired = 0 if negative else int(rng.choice([1, 2, 3], p=[0.60, 0.33, 0.07]))
    for i in range(desired):
        class_id = primary if i == 0 else int(rng.integers(14))
        obj = render_panel(scene, glyph_mask, panel_mask, glyphs, config, split, rng,
                           camera, class_id, len(objects)+1, reserved)
        if obj is None:
            if not objects:
                raise RuntimeError("Cannot place a readable simulated panel")
            continue
        objects.append(obj)
    image, optics = camera_image(scene, rng)
    return image, glyph_mask, panel_mask, objects, camera, optics


def draw_annotations(image, objects):
    result = image.copy()
    for obj in objects:
        corners = np.round(obj["panel_corners_xy"]).astype(np.int32)
        cv2.polylines(result, [corners], True, (80, 230, 90), 1)
        for index, point in enumerate(corners):
            cv2.circle(result, tuple(point), 3, (80, 230, 90), -1)
            cv2.putText(result, str(index), tuple(point+[3, -3]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.30, (80, 230, 90), 1)
        x0, y0, x1, y1 = obj["digit_bbox_xyxy"]
        cv2.rectangle(result, (x0, y0), (x1-1, y1-1), (0, 220, 255), 1)
        cv2.putText(result, obj["class_name"], (x0, max(12, y0-5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (0, 220, 255), 1)
    return result


def yolo_box(box, size):
    x0, y0, x1, y1 = box
    return f"{(x0+x1)/(2*size):.8f} {(y0+y1)/(2*size):.8f} {(x1-x0)/size:.8f} {(y1-y0)/size:.8f}"


def generate(args):
    cv2.setNumThreads(1)
    config = json.loads(Path(args.config).read_text())
    check_config(config)
    if args.size < 320 or min(args.train, args.val, args.test) < 15:
        raise ValueError("Image size must be >=320 and each split must have >=15 images")
    reference, output = Path(args.reference).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    glyphs, provenance = reconstruct_glyphs(reference, output)
    source_copy = output / "reference" / "source_templates"
    source_copy.mkdir(parents=True)
    shutil.copyfile(reference / "templates.json", source_copy / "templates.json")
    shutil.copytree(reference / "crops", source_copy / "crops")
    summaries, previews, closeups = {}, [], {}
    with (output / "manifest.jsonl").open("w") as manifest, \
            (output / "classification_manifest.jsonl").open("w") as crop_manifest:
        for split_index, split in enumerate(SPLITS):
            rng = np.random.default_rng(args.seed+split_index*1000003)
            for folder in ("images", "labels", "masks", "panel_pose/images", "panel_pose/labels", "panel_pose/masks"):
                (output / folder / split).mkdir(parents=True, exist_ok=True)
            for cls in [str(i) for i in range(1, 8)]+["negative"]:
                (output / "classification" / split / cls).mkdir(parents=True, exist_ok=True)
            class_counts, crop_counts, rows_used = Counter(), Counter(), Counter()
            negatives, object_count = 0, 0
            for index in range(getattr(args, split)):
                negative = index % 15 == 14
                image, glyph_mask, panel_mask, objects, camera, optics = make_scene(
                    rng, args.size, glyphs, config, split, index % 15, negative)
                stem = f"{split}_{index:06d}"
                rel_image = f"images/{split}/{stem}.jpg"
                write_image(output / rel_image, image, optics["jpeg_quality"])
                os.link(output / rel_image, output / "panel_pose" / rel_image)
                saved_image = cv2.imread(str(output / rel_image))
                write_image(output / "masks" / split / (stem+".png"), glyph_mask)
                write_image(output / "panel_pose" / "masks" / split / (stem+".png"), panel_mask)
                digit_labels, pose_labels = [], []
                for obj in objects:
                    digit_labels.append(f"{obj['class_id']} {yolo_box(obj['digit_bbox_xyxy'], args.size)}")
                    kpts = " ".join(f"{x/args.size:.8f} {y/args.size:.8f} 2" for x, y in obj["panel_corners_xy"])
                    pose_labels.append(f"0 {yolo_box(obj['panel_bbox_xyxy'], args.size)} {kpts}")
                    crop = classifier_crop(saved_image, obj["digit_bbox_xyxy"])
                    crop_path = f"classification/{split}/{obj['digit']}/{stem}_{obj['instance_id']}.png"
                    write_image(output / crop_path, crop)
                    obj["classification_crop"] = crop_path
                    crop_manifest.write(json.dumps(dict(path=crop_path, split=split, label=str(obj["digit"]),
                        parent_image=rel_image, instance_id=obj["instance_id"], color=obj["color"]))+"\n")
                    class_counts[obj["class_name"]] += 1
                    crop_counts[str(obj["digit"])] += 1
                    rows_used[str(obj["appearance"]["led_rows"])] += 1
                    object_count += 1
                    if obj["class_name"] not in closeups:
                        closeups[obj["class_name"]] = (crop, obj["class_name"])
                if negative:
                    negatives += 1
                    x, y = (int(v) for v in rng.integers(0, args.size-64, 2))
                    crop_path = f"classification/{split}/negative/{stem}.png"
                    write_image(output / crop_path, saved_image[y:y+64, x:x+64])
                    crop_manifest.write(json.dumps(dict(path=crop_path, split=split, label="negative",
                        parent_image=rel_image, instance_id=None))+"\n")
                    crop_counts["negative"] += 1
                for folder, labels in (("labels", digit_labels), ("panel_pose/labels", pose_labels)):
                    (output / folder / split / (stem+".txt")).write_text("\n".join(labels)+("\n" if labels else ""))
                manifest.write(json.dumps(dict(image=rel_image, split=split, index=index,
                    synthetic=True, renderer="procedural_led_planar_3d", camera_matrix=camera.tolist(),
                    image_size=[args.size, args.size], optics=optics, objects=objects))+"\n")
                if split == "train" and (index < 15 or index in (20, 30, 40, 50, 60)):
                    previews.append((draw_annotations(saved_image, objects), stem))
                if (index+1) % 100 == 0 or index+1 == getattr(args, split):
                    print(f"{split}: {index+1}/{getattr(args, split)}", flush=True)
            summaries[split] = dict(images=getattr(args, split), objects=object_count,
                negative_images=negatives, by_class=dict(class_counts), led_rows=dict(rows_used),
                classification_counts=dict(crop_counts))
    contact_sheet(previews, output / "preview.jpg", columns=5, tile_size=320)
    contact_sheet([closeups[name] for name in NAMES], output / "digit_preview.jpg", columns=7)
    common = "train: images/train\nval: images/val\ntest: images/test\n"
    pose_class = "small_panel" if config.get("panel_type") == "small" else "hypothetical_panel"
    (output / "data.yaml").write_text(f"path: {json.dumps(str(output))}\n"+common+"names:\n"+
        "".join(f"  {i}: {name}\n" for i, name in enumerate(NAMES)))
    (output / "panel_pose" / "data.yaml").write_text(
        f"path: {json.dumps(str(output / 'panel_pose'))}\n"+common+
        f"kpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\nnames:\n  0: {pose_class}\n")
    json_file(output / "assumptions.json", config)
    json_file(output / "summary.json", dict(renderer="procedural_led_planar_3d", seed=args.seed,
        image_size=args.size, reference=str(reference), source_crop_provenance=provenance,
        simulation_config=config, counts=summaries, generator_sha256=sha256_file(__file__),
        panel_type=config.get("panel_type", "unspecified"), panel_pose_class=pose_class,
        dependencies=dict(opencv=cv2.__version__, numpy=np.__version__),
        no_recorded_rgb_patches_pasted=True, geometry_status="hypothesis_not_measured",
        real_world_accuracy="not_evaluated"))
    shutil.copyfile(Path(__file__).with_name("SIMULATION.md"), output / "README.md")
    generator = output / "generator"
    generator.mkdir()
    for name in ("simulate.py", "validate_simulation.py", "preview_simulation.py", "generate.py", "SIMULATION.md"):
        shutil.copyfile(Path(__file__).with_name(name), generator / name)
    shutil.copyfile(args.config, generator / "simulation_config.json")
    print(f"Simulation dataset: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="out/armor_digits_v1/reference")
    parser.add_argument("--config", default=str(Path(__file__).with_name("simulation_config.json")))
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", type=int, default=2400)
    parser.add_argument("--val", type=int, default=300)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20261108)
    generate(parser.parse_args())
