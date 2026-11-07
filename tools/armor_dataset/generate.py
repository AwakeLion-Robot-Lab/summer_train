#!/usr/bin/env python3
"""Build synthetic glowing-digit data from reviewed video frames, without inventing plate geometry."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

import cv2
import numpy as np


NAMES = [f"{color}_{digit}" for color in ("red", "blue") for digit in range(1, 8)]
SPLITS = ("train", "val", "test")


def write_image(path, image, jpeg_quality=92):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if path.suffix == ".jpg" else []
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"Cannot write {path}")


def json_file(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounds(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def extract_templates(video, config, output):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if actual != (config["width"], config["height"]):
        raise ValueError(f"Video dimensions differ from reviewed source: {actual}")
    if abs(cap.get(cv2.CAP_PROP_FPS) - config["fps"]) > 0.01:
        raise ValueError("Video frame rate differs from reviewed source")
    rx0, ry0, rx1, ry1 = config["roi_xyxy"]
    templates = []
    for sample in config["samples"]:
        for frame_id in sample["frames"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode reviewed frame {frame_id}")
            crop = frame[ry0:ry1, rx0:rx1].astype(np.float32)
            target, opposite = (2, 0) if sample["color"] == "red" else (0, 2)
            signal = np.maximum(crop[:, :, target] - crop[:, :, opposite], 0)
            if signal.max() < 50:
                raise ValueError(f"No clear colored digit at frame {frame_id}")
            # Identify the luminous core; do not turn the halo into a geometric boundary.
            binary = (signal >= 0.30 * signal.max()).astype(np.uint8)
            joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
            if n <= 1:
                raise ValueError(f"No connected glyph at frame {frame_id}")
            component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            support = ((labels == component) & (binary != 0)).astype(np.uint8)
            box = bounds(support)
            x0, y0, x1, y1 = box
            if y1 - y0 < 30 or x1 - x0 < 10 or (support.sum() / ((x1-x0)*(y1-y0))) < 0.10:
                raise ValueError(f"Implausible glyph at frame {frame_id}: {box}")
            margin = 10
            cx0, cy0 = max(0, x0-margin), max(0, y0-margin)
            cx1, cy1 = min(crop.shape[1], x1+margin), min(crop.shape[0], y1+margin)
            patch = crop[cy0:cy1, cx0:cx1]
            baseline = np.percentile(crop.reshape(-1, 3), 15, axis=0)
            patch = np.maximum(patch - baseline, 0)
            patch_signal = signal[cy0:cy1, cx0:cx1]
            patch[patch_signal < 0.035 * signal.max()] = 0
            mask = support[cy0:cy1, cx0:cx1]
            class_id = (0 if sample["color"] == "red" else 7) + sample["digit"] - 1
            stem = f"{NAMES[class_id]}_f{frame_id:04d}"
            original = frame[ry0+cy0:ry0+cy1, rx0+cx0:rx0+cx1]
            write_image(output / "reference" / "crops" / f"{stem}.png", original)
            write_image(output / "reference" / "masks" / f"{stem}.png", mask * 255)
            info = dict(id=stem, class_id=class_id, digit=sample["digit"], color=sample["color"],
                        frame_index=frame_id, time_s=frame_id/config["fps"],
                        source_bbox_xyxy=[rx0+x0, ry0+y0, rx0+x1, ry0+y1],
                        source_crop_xyxy=[rx0+cx0, ry0+cy0, rx0+cx1, ry0+cy1])
            templates.append(dict(info=info, emission=patch, mask=mask, original=original))
    cap.release()
    if set(t["info"]["class_id"] for t in templates) != set(range(14)):
        raise ValueError("Reviewed source must cover all 14 color/digit combinations")
    json_file(output / "reference" / "templates.json", [t["info"] for t in templates])
    return templates


def contact_sheet(tiles, output, columns=7, tile_size=180):
    rows = math.ceil(len(tiles) / columns)
    sheet = np.full((rows*tile_size, columns*tile_size, 3), 22, np.uint8)
    for i, (image, title) in enumerate(tiles):
        h, w = image.shape[:2]
        scale = min((tile_size-12)/w, (tile_size-30)/h)
        thumb = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))))
        x = i % columns * tile_size
        y = i // columns * tile_size
        tx, ty = x+(tile_size-thumb.shape[1])//2, y+26
        sheet[ty:ty+thumb.shape[0], tx:tx+thumb.shape[1]] = thumb
        cv2.putText(sheet, title, (x+5, y+17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
    write_image(output, sheet)


def background(rng, size, photos):
    source = "procedural"
    if photos and rng.random() < 0.65:
        path = photos[int(rng.integers(len(photos)))]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read background {path}")
        h, w = image.shape[:2]
        side = min(h, w)
        x, y = int(rng.integers(w-side+1)), int(rng.integers(h-side+1))
        image = cv2.resize(image[y:y+side, x:x+side], (size, size)).astype(np.float32)
        image *= rng.uniform(0.3, 0.85)
        source = str(path.resolve())
    else:
        # Generic surfaces and nonnumeric clutter, not a reconstruction of the unseen plate.
        level = rng.uniform(2, 14) if rng.random() < 0.65 else rng.uniform(18, 65)
        coarse = rng.normal(0, 4, (12, 12)).astype(np.float32)
        texture = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
        image = np.repeat((level + texture)[:, :, None], 3, axis=2)
        image *= rng.uniform(0.85, 1.15, (1, 1, 3)).astype(np.float32)
        for _ in range(int(rng.integers(8, 25))):
            x0, y0 = rng.integers(0, size, 2)
            x1, y1 = np.minimum(size-1, [x0, y0] + rng.integers(10, max(11, size//3), 2))
            shade = float(rng.uniform(0, max(8, level*1.5)))
            color = tuple(float(v) for v in shade*rng.uniform(0.85, 1.15, 3))
            cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), color, -1)
        for _ in range(int(rng.integers(2, 10))):
            center = tuple(int(v) for v in rng.integers(0, size, 2))
            radius = int(rng.integers(2, max(4, size//60)))
            color = tuple(float(v) for v in rng.uniform(10, 100, 3))
            cv2.circle(image, center, radius, color, -1)
        image = cv2.GaussianBlur(image, (0, 0), float(rng.uniform(0.6, 2.5)))
    return np.clip(image, 0, 255), source


def render_glyph(template, rng, height):
    emission = template["emission"].copy()
    support = template["mask"]
    x0, y0, x1, y1 = bounds(support)
    # Approximate scan/PWM artefacts, explicitly not measured display timing.
    scan_period = 0
    if rng.random() < 0.25:
        scan_period = int(rng.integers(4, 13))
        start = int(rng.integers(scan_period))
        emission[start::scan_period] *= rng.uniform(0.55, 0.90)
    if rng.random() < 0.15:
        dropout = rng.random(support.shape) < rng.uniform(0.002, 0.015)
        emission[dropout] *= 0.5
    scale = height / (y1-y0)
    squeeze = float(rng.uniform(0.60, 1.12))
    h, w = support.shape
    dw, dh = w*scale*squeeze, h*scale
    src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    dst = np.float32([[-dw/2, -dh/2], [dw/2, -dh/2], [dw/2, dh/2], [-dw/2, dh/2]])
    dst += (rng.uniform(-0.055, 0.055, (4, 2)) * [dw, dh]).astype(np.float32)
    angle = float(rng.uniform(-24, 24))
    theta = np.deg2rad(angle)
    rotation = np.float32([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    dst = dst @ rotation.T
    lo, hi = dst.min(axis=0), dst.max(axis=0)
    pad = max(5, round(height*0.10))
    dst += pad-lo
    out_w, out_h = np.ceil(hi-lo+2*pad+1).astype(int)
    matrix = cv2.getPerspectiveTransform(src, dst)
    patch = cv2.warpPerspective(emission, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(support, matrix, (out_w, out_h), flags=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() < 12:
        raise ValueError("Rendered glyph is too small")
    gain = float(rng.uniform(0.55, 1.35))
    patch *= gain * rng.uniform(0.90, 1.10, (1, 1, 3)).astype(np.float32)
    blur = float(rng.uniform(0.15, max(0.3, min(1.2, height*0.015))))
    patch = cv2.GaussianBlur(patch, (0, 0), blur)
    motion = 1
    if rng.random() < 0.22:
        motion = int(rng.integers(2, max(3, min(6, int(height*0.04)+2))))
        kernel = np.zeros((motion, motion), np.float32)
        if rng.random() < 0.5:
            kernel[motion//2] = 1/motion
        else:
            kernel[:, motion//2] = 1/motion
        patch = cv2.filter2D(patch, -1, kernel)
    glow_sigma = float(rng.uniform(0.8, max(1.0, height*0.055)))
    patch += cv2.GaussianBlur(patch, (0, 0), glow_sigma) * rng.uniform(0.10, 0.40)
    return patch, mask, dict(requested_height_px=height, horizontal_scale=squeeze,
                             rotation_deg=angle, gain=gain, blur_sigma_px=blur,
                             motion_kernel_px=motion, scan_period_source_rows=scan_period,
                             glow_sigma_px=glow_sigma)


def make_scene(rng, size, templates, photos, primary, negative):
    scene, background_source = background(rng, size, photos)
    instances = np.zeros((size, size), np.uint8)
    reserved = np.zeros_like(instances)
    objects = []
    desired = 0 if negative else int(rng.choice([1, 2, 3, 4], p=[0.45, 0.35, 0.15, 0.05]))
    for obj_index in range(desired):
        class_id = primary if obj_index == 0 else int(rng.integers(14))
        pool = templates[class_id]
        template = pool[int(rng.integers(len(pool)))]
        placed = False
        for _ in range(40):
            height = float(np.exp(rng.uniform(np.log(20), np.log(min(135, size*0.30)))) )
            patch, mask, augmentation = render_glyph(template, rng, height)
            h, w = mask.shape
            if h+4 >= size or w+4 >= size:
                continue
            x, y = int(rng.integers(2, size-w-1)), int(rng.integers(2, size-h-1))
            if reserved[y:y+h, x:x+w].any():
                continue
            scene[y:y+h, x:x+w] += patch
            instances[y:y+h, x:x+w][mask > 0] = len(objects)+1
            reserved[max(0,y-4):min(size,y+h+4), max(0,x-4):min(size,x+w+4)] = 1
            local = bounds(mask)
            box = [local[0]+x, local[1]+y, local[2]+x, local[3]+y]
            objects.append(dict(instance_id=len(objects)+1, class_id=class_id,
                                class_name=NAMES[class_id], digit=template["info"]["digit"],
                                color=template["info"]["color"], bbox_xyxy=box,
                                source_template=template["info"]["id"], augmentation=augmentation))
            placed = True
            break
        if not placed and obj_index == 0:
            raise RuntimeError("Unable to place primary glyph")
    noise_sigma = float(rng.uniform(0.3, 2.8))
    noise = rng.normal(0, noise_sigma, (size, size, 1)).astype(np.float32)
    scene = np.clip(scene+noise, 0, 255).astype(np.uint8)
    quality = int(rng.integers(76, 97))
    # Crops and detection images use the same encoded camera-like scene.
    ok, encoded = cv2.imencode(".jpg", scene, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    scene = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return scene, instances, objects, dict(background=background_source, noise_sigma=noise_sigma,
                                         jpeg_quality=quality)


def classifier_crop(image, box, size=64):
    x0, y0, x1, y1 = box
    side = max(x1-x0, y1-y0) * 1.35
    matrix = np.float32([[size/side, 0, size/2-(x0+x1)/2*size/side],
                         [0, size/side, size/2-(y0+y1)/2*size/side]])
    return cv2.warpAffine(image, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)


def annotated(image, objects):
    result = image.copy()
    for obj in objects:
        x0, y0, x1, y1 = obj["bbox_xyxy"]
        color = (80, 220, 255) if obj["color"] == "red" else (255, 220, 80)
        cv2.rectangle(result, (x0, y0), (x1-1, y1-1), color, 1)
        cv2.putText(result, obj["class_name"], (x0, max(12, y0-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
    return result


def generate(args):
    cv2.setNumThreads(1)
    if args.image_size < 256:
        raise ValueError("--image-size must be at least 256")
    if min(args.train, args.val, args.test) < 14:
        raise ValueError("Each split needs at least 14 images for class coverage")
    video, output = Path(args.video).resolve(), Path(args.output).resolve()
    config = json.loads(Path(args.source_config).read_text())
    source_hash = sha256_file(video)
    if source_hash != config["video_sha256"]:
        raise ValueError("Source hash differs: review frames and update source config for this video")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty; existing data will not be overwritten: {output}")
    photos = []
    if args.backgrounds:
        # Only explicit background-negative images are accepted, not screenshots with hidden targets.
        root = Path(args.backgrounds).resolve()
        photos = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
        if len(photos) < 3:
            raise ValueError("Provide at least three target-free background photos, one per split")
    output.mkdir(parents=True, exist_ok=True)
    templates = extract_templates(video, config, output)
    grouped = {k: [t for t in templates if t["info"]["class_id"] == k] for k in range(14)}
    contact_sheet([(t["original"], t["info"]["id"]) for t in templates],
                  output / "reference" / "templates.jpg")
    shutil.copyfile(args.source_config, output / "reference" / "source_config.json")
    counts, classification_counts, preview, class_tiles = {}, {}, [], {}
    with (output / "manifest.jsonl").open("w") as manifest, \
            (output / "classification_manifest.jsonl").open("w") as class_manifest:
        for split_index, split in enumerate(SPLITS):
            count = getattr(args, split)
            seed = args.seed + split_index*1000003
            rng = np.random.default_rng(seed)
            split_photos = photos[split_index::3]
            for folder in ("images", "labels", "masks"):
                (output / folder / split).mkdir(parents=True, exist_ok=True)
            for digit in [str(v) for v in range(1, 8)] + ["negative"]:
                (output / "classification" / split / digit).mkdir(parents=True, exist_ok=True)
            class_count, crop_count, backgrounds, negatives = Counter(), Counter(), Counter(), 0
            # One guaranteed primary instance per color/digit in each group of 15 images.
            for i in range(count):
                negative = i % 15 == 14
                primary = i % 15
                scene, instances, objects, scene_params = make_scene(
                    rng, args.image_size, grouped, split_photos, primary, negative)
                stem = f"{split}_{i:06d}"
                rel_image = f"images/{split}/{stem}.jpg"
                rel_mask = f"masks/{split}/{stem}.png"
                write_image(output / rel_image, scene, scene_params["jpeg_quality"])
                # Read back once so classifier crops reflect the actual saved detector image.
                saved_scene = cv2.imread(str(output / rel_image))
                write_image(output / rel_mask, instances)
                labels = []
                for obj in objects:
                    x0, y0, x1, y1 = obj["bbox_xyxy"]
                    s = args.image_size
                    labels.append(f"{obj['class_id']} {(x0+x1)/(2*s):.8f} {(y0+y1)/(2*s):.8f} "
                                  f"{(x1-x0)/s:.8f} {(y1-y0)/s:.8f}")
                    class_count[obj["class_name"]] += 1
                    crop = classifier_crop(saved_scene, obj["bbox_xyxy"])
                    crop_path = f"classification/{split}/{obj['digit']}/{stem}_{obj['instance_id']}.png"
                    write_image(output / crop_path, crop)
                    obj["classification_crop"] = crop_path
                    crop_count[str(obj["digit"])] += 1
                    class_manifest.write(json.dumps(dict(path=crop_path, split=split, label=str(obj["digit"]),
                        color=obj["color"], parent_image=rel_image, instance_id=obj["instance_id"],
                        source_template=obj["source_template"])) + "\n")
                    if obj["class_name"] not in class_tiles:
                        class_tiles[obj["class_name"]] = (crop, obj["class_name"])
                if negative:
                    negatives += 1
                    x, y = (int(v) for v in rng.integers(0, args.image_size-64, 2))
                    crop_path = f"classification/{split}/negative/{stem}.png"
                    write_image(output / crop_path, saved_scene[y:y+64, x:x+64])
                    crop_count["negative"] += 1
                    class_manifest.write(json.dumps(dict(path=crop_path, split=split, label="negative",
                        parent_image=rel_image, instance_id=None, source_template=None)) + "\n")
                (output / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))
                record = dict(image=rel_image, mask=rel_mask, split=split, index=i, split_seed=seed,
                              size=[args.image_size, args.image_size], synthetic=True,
                              scene=scene_params, objects=objects)
                manifest.write(json.dumps(record) + "\n")
                backgrounds[scene_params["background"]] += 1
                if split == "train" and (i < 14 or i == 14 or i in (30, 40, 50, 60, 70, 80)):
                    preview.append((annotated(saved_scene, objects), stem))
                if (i+1) % 100 == 0 or i+1 == count:
                    print(f"{split}: {i+1}/{count}", flush=True)
            counts[split] = dict(images=count, objects=sum(class_count.values()), negative_images=negatives,
                                 by_class=dict(sorted(class_count.items())), backgrounds=dict(backgrounds))
            classification_counts[split] = dict(sorted(crop_count.items()))
    contact_sheet(preview, output / "preview.jpg", columns=5, tile_size=320)
    contact_sheet([class_tiles[name] for name in NAMES], output / "classification_preview.jpg", tile_size=180)
    yaml = f"path: {json.dumps(str(output))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml += "".join(f"  {i}: {name}\n" for i, name in enumerate(NAMES))
    (output / "data.yaml").write_text(yaml)
    json_file(output / "classes.json", dict(detection={i: name for i, name in enumerate(NAMES)},
        classification={i: str(i+1) for i in range(7)} | {7: "negative"},
        robot_role_mapping=None, physical_keypoints=None))
    json_file(output / "summary.json", dict(source_video=str(video), source_sha256=source_hash,
        source_link="https://v.douyin.com/FsP4WOCBv8c/", seed=args.seed, image_size=args.image_size,
        source_templates=len(templates), counts=counts, classification_counts=classification_counts,
        dependencies=dict(opencv=cv2.__version__, numpy=np.__version__),
        generator_sha256=sha256_file(__file__), source_config=config,
        scope="Luminous glyph detection and classification only; not physical armor localization.",
        split_limit="All splits share glyph templates from one video; synthetic validation/test do not measure real-world generalization.",
        geometry_limit="No measured dimensions, physical screen corners, LED pitch, or measured refresh timing.",
        source_limit="Red digit 6 has only one reviewed source frame; blue digit 6 has two."))
    shutil.copyfile(Path(__file__).with_name("README.md"), output / "README.md")
    generator_dir = output / "generator"
    generator_dir.mkdir()
    for name in ("generate.py", "validate.py", "README.md"):
        shutil.copyfile(Path(__file__).with_name(name), generator_dir / name)
    shutil.copyfile(args.source_config, generator_dir / "source_video.json")
    print(f"Generated dataset: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--source-config", default=str(Path(__file__).with_name("source_video.json")))
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", type=int, default=1400)
    parser.add_argument("--val", type=int, default=280)
    parser.add_argument("--test", type=int, default=280)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--backgrounds", help="Optional directory of target-free photos; partitioned by split")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
