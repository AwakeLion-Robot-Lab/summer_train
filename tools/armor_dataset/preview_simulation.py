#!/usr/bin/env python3
"""Render a continuous motion demonstration from a generated v2 dataset."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from simulate import arena_background, camera_image, draw_annotations, render_panel


def make_preview(dataset, output=None):
    cv2.setNumThreads(1)
    root = Path(dataset).resolve()
    config = json.loads((root / "assumptions.json").read_text())
    glyphs = {i: (cv2.imread(str(root / "reference" / "silhouettes" / f"{i}.png"), 0) > 0).astype(np.uint8)
              for i in range(1, 8)}
    size = json.loads((root / "summary.json").read_text())["image_size"]
    camera = np.array([[1.3*size, 0, size/2], [0, 1.3*size, size/2], [0, 0, 1]], float)
    output = Path(output) if output else root / "simulation_preview.mp4"
    if output.exists():
        raise ValueError(f"Preview already exists: {output}")
    fps = 24
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot open the MP4 writer")
    for index in range(7*fps):
        time = index/fps
        scene = arena_background(np.random.default_rng(57), size)
        gm = np.zeros((size, size), np.uint8)
        pm, occupied = gm.copy(), gm.copy()
        objects = []
        for side in range(2):
            phase = time*1.2+side*1.7
            u = size*(0.27+0.46*side+0.02*np.sin(phase))
            v = size*(0.53+0.04*np.sin(phase*0.8))
            pose = (38*np.sin(phase), 12*np.cos(phase*0.9), 8*np.sin(phase*0.7),
                    u, v, 0.88+0.10*np.sin(phase*0.6))
            digit_class = int(time) % 7 + side*7
            obj = render_panel(scene, gm, pm, glyphs, config, "train", np.random.default_rng(101+side),
                               camera, digit_class, side+1, occupied, fixed_pose=pose)
            if obj is None:
                raise RuntimeError("Cannot place preview panels")
            objects.append(obj)
        image, _ = camera_image(scene, np.random.default_rng(103))
        image = draw_annotations(image, objects)
        title = ("SIMULATION - small panel; display layout unverified" if config.get("panel_type") == "small"
                 else "SIMULATION - hypothetical panel geometry")
        basis = "project geometry" if config.get("geometry_source") else "assumed"
        cv2.putText(image, title, (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1)
        cv2.putText(image, f"{config['panel_width_m']*1000:.0f} x {config['panel_height_m']*1000:.0f} mm ({basis})",
                    (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
        writer.write(image)
    writer.release()
    print(f"Preview: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--output")
    args = parser.parse_args()
    make_preview(args.dataset, args.output)
