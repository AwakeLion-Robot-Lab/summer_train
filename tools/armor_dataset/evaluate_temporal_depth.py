#!/usr/bin/env python3
"""Causal replay with per-instance states, real recording timestamps and separate lag benchmarks."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from depth_data import geometry_features, read_records, teacher_metrics
from temporal_depth import DepthFilterConfig, DepthKalmanFilter, DigitBoxAssociator


def json_file(path,value):
    path.write_text(json.dumps(value,indent=2)+"\n")


def recording_times(rows,mode):
    source=Path(rows[0]["source_video"]).with_suffix(".txt")
    if mode!="nominal" and source.exists():
        times=np.array([float(line.split()[0]) for line in source.read_text().splitlines() if line.strip()])
        if not np.isfinite(times).all() or not np.all(np.diff(times)>0):
            raise ValueError(f"Invalid paired recording timestamps: {source}")
        if max(r["frame_index"] for r in rows)>=len(times):
            raise ValueError(f"Timestamp text ends before selected frame: {source}")
        info=dict(source="paired_recording_text",path=str(source),
                  sha256=hashlib.sha256(source.read_bytes()).hexdigest(),median_frame_interval_s=float(np.median(np.diff(times))))
        return {r["id"]:float(times[r["frame_index"]]) for r in rows},info
    if mode=="paired":raise ValueError(f"Missing paired timestamps: {source}")
    return {r["id"]:r["time_s_nominal"] for r in rows},dict(source="nominal_video_fps",latency_precision="nominal_only")


def synthetic_benchmarks(config,output):
    fps=30;t=np.arange(300)/fps;rng=np.random.default_rng(51)
    def run(raw):
        f=DepthKalmanFilter(config)
        return [f.update(float(z),float(s)) for s,z in zip(t,raw)]
    stationary=3+rng.normal(0,.05,len(t))
    stable=run(stationary)
    clean=3+.30*np.sin(2*np.pi*.5*t)
    noisy=clean+rng.normal(0,.03,len(t))
    motion=run(noisy)
    y=np.array([r["depth_z_m"] for r in motion])
    active=(t>=1)&(t<=t[-1]-.30)
    lags=np.linspace(-.10,.30,121)
    costs=[np.mean((y[active]-np.interp(t[active]-lag,t,clean))**2) for lag in lags]
    lag=float(lags[int(np.argmin(costs))])
    step=np.full(len(t),3.0);step[t>=3]=3.6
    stepped=run(step)
    crossing=next((float(s) for s,r in zip(t,stepped) if s>=3 and r["depth_z_m"] is not None and r["depth_z_m"]>=3.3),None)
    raw_std=float(np.std(stationary[t>=1]-3))
    filtered_std=float(np.std([r["depth_z_m"] for s,r in zip(t,stable) if s>=1]))
    report=dict(source="deterministic_synthetic_truth_not_real_world_accuracy",fps=fps,
        stationary_raw_std_m=raw_std,stationary_filtered_std_m=filtered_std,
        stationary_std_reduction_fraction=1-filtered_std/raw_std,
        sinusoid_frequency_hz=.5,sinusoid_best_alignment_lag_s=lag,
        sinusoid_raw_mae_m=float(np.mean(np.abs(noisy[active]-clean[active]))),
        sinusoid_filtered_mae_m=float(np.mean(np.abs(y[active]-clean[active]))),
        step_half_response_delay_s=None if crossing is None else crossing-3,
        notes="Alignment lag is an empirical motion-response estimate. It is not filter CPU time or live camera-to-output latency.")
    with (output/"synthetic_traces.jsonl").open("w") as stream:
        for i,s in enumerate(t):
            stream.write(json.dumps(dict(time_s=float(s),stationary_raw=float(stationary[i]),
                stationary_filtered=stable[i]["depth_z_m"],motion_truth=float(clean[i]),
                motion_raw=float(noisy[i]),motion_filtered=motion[i]["depth_z_m"],
                step_truth=float(step[i]),step_filtered=stepped[i]["depth_z_m"]))+"\n")
    return report


def summarize(records):
    valid=[r for r in records if r["filtered_depth_z_m"] is not None]
    if not valid:return dict(observations=len(records),valid_outputs=0)
    truth=np.array([r["teacher_depth_z_m"] for r in valid])
    raw=np.array([r["raw_depth_z_m"] for r in valid])
    filtered=np.array([r["filtered_depth_z_m"] for r in valid])
    raw_steps,filtered_steps=[],[]
    previous={}
    for r in records:
        p=previous.get(r["track_id"])
        if (p is not None and r["timestamp_s"]-p["timestamp_s"]<=.15 and r["filtered_depth_z_m"] is not None
                and p["filtered_depth_z_m"] is not None):
            raw_steps.append(r["raw_depth_z_m"]-p["raw_depth_z_m"])
            filtered_steps.append(r["filtered_depth_z_m"]-p["filtered_depth_z_m"])
        previous[r["track_id"]]=r
    rms=lambda values:float(np.sqrt(np.mean(np.square(values)))) if values else None
    return dict(observations=len(records),valid_outputs=len(valid),invalid_outputs=len(records)-len(valid),
        raw_vs_teacher_on_valid_frames=teacher_metrics(truth,raw),
        filtered_vs_teacher=teacher_metrics(truth,filtered),
        adjacent_raw_variation_rms_m=rms(raw_steps),adjacent_filtered_variation_rms_m=rms(filtered_steps),
        adjacent_pairs=len(raw_steps),variation_note="Includes target motion; this metric is not pure measurement noise.",
        filter_status_counts=dict(Counter(r["filter"]["status"] for r in records)),
        filter_compute_ms_p50_p95=np.percentile([r["filter"]["filter_compute_ms"] for r in records],[50,95]).tolist())


def evaluate(args):
    root,model,output=Path(args.dataset),Path(args.model),Path(args.output)
    if output.exists() and any(output.iterdir()):raise ValueError("Output must be empty")
    config=DepthFilterConfig(**json.loads(Path(args.filter_config).read_text()))
    model_config=json.loads((model/"model_config.json").read_text())
    method=model_config["recommended_method"] if args.method=="selected" else args.method
    rows=read_records(root)
    predictions={r["id"]:r["predicted_depth_z_m"] for r in
                 map(json.loads,(model/"predictions.jsonl").read_text().splitlines())}
    output.mkdir(parents=True,exist_ok=True)
    rendered,timestamp_sources=[],{}
    for sequence in sorted({r["sequence"] for r in rows}):
        group=sorted([r for r in rows if r["sequence"]==sequence],key=lambda r:(r["frame_index"],r["id"]))
        times,origin=recording_times(group,args.timestamps)
        timestamp_sources[sequence]=origin
        tracker=DigitBoxAssociator(config.max_gap_s)
        filters={}
        for frame in sorted({r["frame_index"] for r in group}):
            frame_rows=[r for r in group if r["frame_index"]==frame]
            now=times[frame_rows[0]["id"]]
            identities=tracker.update([r["digit_bbox_xyxy"] for r in frame_rows],now)
            for row,identity in zip(frame_rows,identities):
                if method=="network":
                    raw=predictions[row["id"]]
                else:
                    geometry=geometry_features(row["digit_bbox_xyxy"],row["camera_matrix"])
                    raw=float(np.exp(model_config["log_height_prior"]-float(geometry[1])))
                if identity not in filters:filters[identity]=DepthKalmanFilter(config)
                filtered=filters[identity].update(raw,now)
                rendered.append(dict(id=row["id"],sequence=sequence,split=row["split"],frame_index=frame,
                    timestamp_s=now,track_id=f"{sequence}:{identity}",bbox=row["digit_bbox_xyxy"],
                    raw_depth_z_m=raw,filtered_depth_z_m=filtered["depth_z_m"],
                    teacher_depth_z_m=row["depth_z_m"],filter=filtered))
    (output/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rendered))
    synthetic=synthetic_benchmarks(config,output)
    report=dict(method=method,filter_config=asdict(config),timestamp_sources=timestamp_sources,
        splits={split:summarize([r for r in rendered if r["split"]==split]) for split in ("train","val","test")},
        synthetic=synthetic,parameters_selected_using_test=False,
        limitations=["Real recording depth references are PnP pseudo labels, not independent truth.",
                     "ROI and sparse observation availability come from the old teacher exporter.",
                     "Offline association uses only boxes; instance identities are not ground truth.",
                     "No live camera-to-output delay was measured; compute cost and synthetic lag are separate."])
    json_file(output/"report.json",report)
    try:
        os.environ.setdefault("MPLCONFIGDIR",str(output.resolve()/".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        traces=[json.loads(l) for l in (output/"synthetic_traces.jsonl").read_text().splitlines()]
        fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
        for ax,(key,title) in zip(axes,[("stationary","Stationary noise"),("motion","Moving target"),("step","Actual distance step")]):
            x=[r["time_s"] for r in traces]
            if key+"_raw" in traces[0]:ax.plot(x,[r[key+"_raw"] for r in traces],alpha=.4,label="raw")
            truth=[3.0 for r in traces] if key=="stationary" else [r[key+"_truth"] for r in traces]
            ax.plot(x,truth,"k--",label="synthetic truth")
            ax.plot(x,[r[key+"_filtered"] for r in traces],label="filtered")
            ax.set_ylabel("Z (m)");ax.set_title(title);ax.legend(loc="upper right")
        axes[-1].set_xlabel("Capture time (s)");fig.tight_layout();fig.savefig(output/"synthetic_response.png",dpi=140);plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",default="out/armor_sticker_depth_v2")
    parser.add_argument("--model",default="out/armor_sticker_depth_model_v2")
    parser.add_argument("--output",required=True)
    parser.add_argument("--filter-config",default=str(Path(__file__).with_name("depth_filter_config.json")))
    parser.add_argument("--method",choices=["selected","network","inverse_digit_height_baseline"],default="selected")
    parser.add_argument("--timestamps",choices=["auto","paired","nominal"],default="auto")
    evaluate(parser.parse_args())
