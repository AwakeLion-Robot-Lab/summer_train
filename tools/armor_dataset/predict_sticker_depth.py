#!/usr/bin/env python3
"""Student-only depth inference from a raw image and a digit detector's bounding box."""

import argparse
import json
import math
from pathlib import Path

import cv2
import torch

from depth_data import geometry_features, glyph_image


def infer(model_dir,image,box,camera):
    model_dir=Path(model_dir)
    config=json.loads((model_dir/"model_config.json").read_text())
    x0,y0,x1,y1=map(int,box)
    if not (0<=x0<x1<=image.shape[1] and 0<=y0<y1<=image.shape[0]):
        raise ValueError("Digit box is outside original image")
    tensor=torch.from_numpy(glyph_image(image[y0:y1,x0:x1])[None])
    geometry=torch.from_numpy(geometry_features(box,camera)[None])
    model=torch.jit.load(str(model_dir/"student.ts"),map_location="cpu").eval()
    with torch.no_grad():depth,logits=model(tensor,geometry)
    network=float(depth.item())
    baseline=math.exp(config["log_height_prior"]-float(geometry[0,1]))
    method=config.get("recommended_method","network")
    selected=network if method=="network" else baseline
    digit=config["class_values"][int(logits.argmax(1))] if config["classification_enabled"] else None
    return dict(depth_z_m=selected,used_method=method,network_depth_z_m=network,
                inverse_digit_height_depth_z_m=baseline,digit=digit,classification_enabled=config["classification_enabled"],
                input="glyph pixels, digit bbox and intrinsics; PnP and lightbars are not called")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--image",required=True)
    parser.add_argument("--bbox",required=True,type=float,nargs=4,metavar=("X0","Y0","X1","Y1"))
    parser.add_argument("--camera",required=True,help="JSON 3x3 camera matrix or calibration YAML")
    args=parser.parse_args()
    image=cv2.imread(args.image)
    if image is None:raise ValueError("Cannot read input image")
    path=Path(args.camera)
    if path.suffix in (".yaml",".yml"):
        import yaml
        node=yaml.safe_load(path.read_text())["calibration"]
        if image.shape[1]!=node["image_width"] or image.shape[0]!=node["image_height"]:
            raise ValueError("Original image size differs from calibration")
        camera=node["camera_matrix"]
    else:
        camera=json.loads(path.read_text())
    print(json.dumps(infer(args.model,image,args.bbox,camera),indent=2))
