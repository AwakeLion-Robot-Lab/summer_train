"""Student inputs: raw glyph pixels and geometry obtainable from a digit detector, never lightbars."""

import json
from pathlib import Path

import cv2
import numpy as np


FEATURE_NAMES = ["log_width_over_fx", "log_height_over_fy", "ray_x", "ray_y", "log_aspect"]


def geometry_features(box, camera):
    x0, y0, x1, y1 = np.asarray(box, float)
    k = np.asarray(camera, float)
    width, height = x1-x0, y1-y0
    if width <= 0 or height <= 0 or k.shape != (3, 3) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid digit box/camera matrix")
    result = np.array([np.log(width/k[0, 0]), np.log(height/k[1, 1]),
                       ((x0+x1)/2-k[0, 2])/k[0, 0], ((y0+y1)/2-k[1, 2])/k[1, 1],
                       np.log(width/height)], np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite student features")
    return result


def glyph_image(image, size=64):
    if image is None or image.ndim != 3:
        raise ValueError("Cannot read digit crop")
    # Printed white strokes remain; colored lightbar spill is suppressed.
    white = image.min(axis=2)
    h, w = white.shape
    scale = (size-8)/max(h, w)
    resized = cv2.resize(white, (max(1, round(w*scale)), max(1, round(h*scale))))
    canvas = np.zeros((size, size), np.float32)
    y, x = (size-resized.shape[0])//2, (size-resized.shape[1])//2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized/255.0
    return canvas[None]


def read_records(root):
    root = Path(root)
    return [json.loads(line) for line in (root/"manifest.jsonl").read_text().splitlines() if line]


def teacher_metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    error = np.abs(prediction-truth)
    return dict(samples=len(truth), mae_m=float(error.mean()), median_absolute_error_m=float(np.median(error)),
                p95_absolute_error_m=float(np.percentile(error, 95)),
                mean_absolute_relative_error=float(np.mean(error/truth)))
