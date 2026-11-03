#!/usr/bin/env python3
"""Compare rune_detect logs using image motion, without treating NN scores as truth.

Requires Python OpenCV and NumPy. Coordinates in logs and metrics use source pixels.
ECC registers each image pair independently; no temporal smoothing is applied.
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def read_log(path):
    frames = {}
    for line in Path(path).read_text().splitlines():
        header = re.match(r"frame=(\d+) detections=(\d+)", line)
        if header:
            index = int(header[1]) - 1
            frames[index] = []
        candidate = re.match(r"  class=(\d+) confidence=([\d.e+-]+)", line)
        if candidate:
            points = np.array(re.findall(r"\(([\d.e+-]+),([\d.e+-]+)\)", line), float)
            valid = line.split("valid=")[1].strip() if "valid=" in line else "11111"
            if points.shape != (5, 2):
                raise ValueError(f"invalid five-point log at frame {index+1}")
            frames[index].append(dict(cls=int(candidate[1]), points=points, valid=valid))
    return frames


def summary(values):
    a = np.array(values)
    if not len(a):
        return dict(n=0)
    return dict(n=len(a), p50=float(np.median(a)), p95=float(np.percentile(a,95)),
                max=float(a.max()), rms=float(np.sqrt(np.mean(a*a))))


def register(previous, current, center, half_size, motion):
    x, y = np.round(center).astype(int)
    x0, y0 = max(0,x-half_size), max(0,y-half_size)
    x1, y1 = min(previous.shape[1],x+half_size), min(previous.shape[0],y+half_size)
    a, b = previous[y0:y1,x0:x1], current[y0:y1,x0:x1]
    if min(a.shape) < 20:
        return None
    warp = np.eye(2,3,dtype=np.float32)
    try:
        correlation, warp = cv2.findTransformECC(a,b,warp,motion,
            (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,1e-5))
    except cv2.error:
        return None
    if correlation < .98 or not np.isfinite(warp).all():
        return None
    if motion == cv2.MOTION_AFFINE:
        scales = np.linalg.svd(warp[:,:2],compute_uv=False)
        if scales.min() < .9 or scales.max() > 1.1 or np.linalg.det(warp[:,:2]) <= 0:
            return None
    # Convert ROI-relative affine translation into full-image coordinates.
    origin = np.array([x0,y0])
    warp[:,2] += origin - warp[:,:2] @ origin
    return warp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',required=True)
    parser.add_argument('--before',required=True)
    parser.add_argument('--after',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    before, after = read_log(args.before), read_log(args.after)
    if not before or before.keys() != after.keys():
        raise ValueError('logs must cover the same nonempty frame range')
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError('cannot open video')
    residuals = {name:{'r':[],'target_center':[],'target_points':[]} for name in ('before','after')}
    corrections = []
    previous = None
    previous_detections = None
    rows = []
    counts = {name:sum(bool(ds) for ds in data.values()) for name,data in [('before',before),('after',after)]}
    for i in range(len(before)):
        ok, image = cap.read()
        if not ok:
            raise ValueError(f'video ended before log at frame {i+1}')
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ds = [next((d for d in data[i] if d['cls']==0 and d['valid']=='11111'),None) for data in (before,after)]
        if all(d is not None for d in ds):
            corrections.append(np.linalg.norm(ds[1]['points']-ds[0]['points'],axis=1))
        if previous is not None and all(d is not None for d in ds+previous_detections):
            a = previous_detections[0]['points']
            b = ds[0]['points']
            center = a[:4].mean(0)
            if np.linalg.norm(b[:4].mean(0)-center) <= 30:
                r_warp = register(previous,gray,(a[4]+b[4])/2,40,cv2.MOTION_TRANSLATION)
                half_size = int(np.ceil(np.linalg.norm(a[:4]-center,axis=1).max()+15))
                target_warp = register(previous,gray,center,half_size,cv2.MOTION_AFFINE)
                row = dict(frame_1based=i+1)
                for name, old, new in zip(('before','after'),previous_detections,ds):
                    pa,pb=old['points'],new['points']
                    if r_warp is not None:
                        predicted=r_warp[:,:2]@pa[4]+r_warp[:,2]
                        residual=float(np.linalg.norm(pb[4]-predicted))
                        residuals[name]['r'].append(residual)
                        row[name+'_r_residual_px']=residual
                    if target_warp is not None:
                        predicted=pa[:4]@target_warp[:,:2].T+target_warp[:,2]
                        residuals[name]['target_points'].extend(np.linalg.norm(pb[:4]-predicted,axis=1).tolist())
                        residual=float(np.linalg.norm(pb[:4].mean(0)-predicted.mean(0)))
                        residuals[name]['target_center'].append(residual)
                        row[name+'_target_center_residual_px']=residual
                rows.append(row)
        previous=gray
        previous_detections=ds
    cap.release()
    report = dict(video=args.video,before_log=args.before,after_log=args.after,total_frames=len(before),
        detected_frames=counts, detection_count_changed_frames=sum(len(before[i])!=len(after[i]) for i in before),
        metrics={name:{key+'_residual_px':summary(v) for key,v in measurements.items()} for name,measurements in residuals.items()},
        coordinate_correction_px=summary(np.array(corrections).reshape(-1)),
        note='Paired image-registration residuals, not annotated absolute localization errors. '
             'Target-switch pairs (>30px), partial detections and ECC correlation <0.98 are excluded.',
        r_worst_pairs=sorted((r for r in rows if 'before_r_residual_px' in r),key=lambda r:-r['before_r_residual_px'])[:12],
        target_worst_pairs=sorted((r for r in rows if 'after_target_center_residual_px' in r),
                                 key=lambda r:-r['after_target_center_residual_px'])[:12])
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('r_worst_pairs','target_worst_pairs')},indent=2))


if __name__=='__main__':
    main()
