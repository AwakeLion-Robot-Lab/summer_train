#!/usr/bin/env python3
"""Student-only depth inference from a raw image and a digit detector's bounding box."""

import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from depth_data import geometry_features, glyph_image
from temporal_depth import DepthFilterBank, DepthFilterConfig


class StickerDepthPredictor:
    """Load once per stream; each tracked instance has an independent causal filter."""

    def __init__(self, model_dir, temporal_config=None):
        model_dir=Path(model_dir)
        self.config=json.loads((model_dir/"model_config.json").read_text())
        self.model=torch.jit.load(str(model_dir/"student.ts"),map_location="cpu").eval()
        if temporal_config is not None and not isinstance(temporal_config,DepthFilterConfig):
            temporal_config=DepthFilterConfig(**temporal_config)
        self.filters=DepthFilterBank(temporal_config) if temporal_config is not None else None
        self.camera_signatures={}

    def reset(self):
        if self.filters is not None:
            self.filters.reset()
        self.camera_signatures.clear()

    def predict(self,image,box,camera,*,timestamp_s=None,target_id=None,sequence_id="default",available_time_s=None):
        if timestamp_s is not None:
            timestamp_s=float(timestamp_s)
            if not math.isfinite(timestamp_s):raise ValueError("Capture timestamp must be finite")
        if self.filters is not None and (timestamp_s is None or target_id is None):
            raise ValueError("Filtered inference requires capture timestamp_s and a stable target_id")
        if available_time_s is not None and (not math.isfinite(available_time_s) or timestamp_s is None or available_time_s<timestamp_s):
            raise ValueError("Available/capture times must use the same clock, with available >= capture")
        started=time.perf_counter()
        config=self.config
        x0,y0,x1,y1=map(int,box)
        if not (0<=x0<x1<=image.shape[1] and 0<=y0<y1<=image.shape[0]):
            raise ValueError("Digit box is outside original image")
        tensor=torch.from_numpy(glyph_image(image[y0:y1,x0:x1])[None])
        geometry=torch.from_numpy(geometry_features(box,camera)[None])
        with torch.no_grad():depth,logits=self.model(tensor,geometry)
        network=float(depth.item())
        baseline=math.exp(config["log_height_prior"]-float(geometry[0,1]))
        method=config.get("recommended_method","network")
        selected=network if method=="network" else baseline
        digit=config["class_values"][int(logits.argmax(1))] if config["classification_enabled"] else None
        elapsed_ms=(time.perf_counter()-started)*1000
        result=dict(depth_z_m=selected,raw_depth_z_m=selected,used_method=method,network_depth_z_m=network,
                    inverse_digit_height_depth_z_m=baseline,digit=digit,classification_enabled=config["classification_enabled"],
                    inference_compute_ms=elapsed_ms,temporal=None,
                    observation_age_s=None if available_time_s is None else available_time_s-timestamp_s,
                    input="glyph pixels, digit bbox and intrinsics; PnP and lightbars are not called")
        if timestamp_s is not None:
            result.update(timestamp_s=timestamp_s,target_id=target_id,sequence_id=sequence_id)
        if self.filters is not None:
            key=(sequence_id,target_id)
            self.filters.prune(float(timestamp_s))
            self.camera_signatures={k:v for k,v in self.camera_signatures.items() if k in self.filters.filters}
            signature=(image.shape[:2],np.asarray(camera,dtype=np.float64).tobytes())
            changed=key in self.camera_signatures and self.camera_signatures[key]!=signature
            if changed:
                self.filters.reset(key)
            self.camera_signatures[key]=signature
            filtered=self.filters.update(key,selected,timestamp_s)
            if changed:
                filtered["status"]="reset_camera"
            result.update(depth_z_m=filtered["depth_z_m"],temporal=filtered,target_id=target_id,
                          sequence_id=sequence_id,timestamp_s=timestamp_s)
        return result

    def missing(self,target_id,timestamp_s,sequence_id="default"):
        if self.filters is None:
            return dict(depth_z_m=None,raw_depth_z_m=None,temporal=None,target_id=target_id,timestamp_s=timestamp_s)
        self.filters.prune(float(timestamp_s))
        filtered=self.filters.update((sequence_id,target_id),None,timestamp_s)
        return dict(depth_z_m=filtered["depth_z_m"],raw_depth_z_m=None,temporal=filtered,
                    target_id=target_id,sequence_id=sequence_id,timestamp_s=timestamp_s)


def infer(model_dir,image,box,camera):
    """Backward-compatible single-image inference; temporal state needs a persistent predictor."""
    return StickerDepthPredictor(model_dir).predict(image,box,camera)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image")
    mode.add_argument("--stream",help="JSONL: image, bbox, timestamp_s, target_id; or missing=true")
    parser.add_argument("--bbox",type=float,nargs=4,metavar=("X0","Y0","X1","Y1"))
    parser.add_argument("--camera",required=True,help="JSON 3x3 camera matrix or calibration YAML")
    parser.add_argument("--filter-config",default=str(Path(__file__).with_name("depth_filter_config.json")))
    parser.add_argument("--no-filter",action="store_true",help="Disable temporal processing in stream mode")
    parser.add_argument("--output",help="Write stream results as JSONL; otherwise print to stdout")
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    path=Path(args.camera)
    expected_size=None
    if path.suffix in (".yaml",".yml"):
        import yaml
        node=yaml.safe_load(path.read_text())["calibration"]
        expected_size=(node["image_width"],node["image_height"])
        camera=node["camera_matrix"]
    else:
        camera=json.loads(path.read_text())
    settings=json.loads(Path(args.filter_config).read_text()) if args.stream and not args.no_filter else None
    predictor=StickerDepthPredictor(args.model,settings)
    def load_image(name):
        image=cv2.imread(str(name))
        if image is None:raise ValueError(f"Cannot read input image: {name}")
        if expected_size is not None and (image.shape[1],image.shape[0])!=expected_size:
            raise ValueError("Original image size differs from calibration")
        return image
    if args.image:
        if args.bbox is None:parser.error("--image requires --bbox")
        print(json.dumps(predictor.predict(load_image(args.image),args.bbox,camera),indent=2))
    else:
        import sys
        stream_path=Path(args.stream).resolve()
        destination=Path(args.output) if args.output else None
        if destination is not None and destination.exists():raise ValueError("Output already exists")
        if destination is not None:destination.parent.mkdir(parents=True,exist_ok=True)
        out=destination.open("w") if destination is not None else sys.stdout
        try:
            for line in stream_path.read_text().splitlines():
                if not line.strip():continue
                row=json.loads(line)
                identity=row["target_id"]
                timestamp=row["timestamp_s"]
                sequence=row.get("sequence_id","default")
                if row.get("missing",False):
                    result=predictor.missing(identity,timestamp,sequence)
                else:
                    image_path=Path(row["image"])
                    if not image_path.is_absolute():image_path=stream_path.parent/image_path
                    result=predictor.predict(load_image(image_path),row["bbox"],camera,timestamp_s=timestamp,
                        target_id=identity,sequence_id=sequence,available_time_s=row.get("available_time_s"))
                out.write(json.dumps(result)+"\n")
                out.flush()
        finally:
            if destination is not None:out.close()
