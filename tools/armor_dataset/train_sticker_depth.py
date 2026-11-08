#!/usr/bin/env python3
"""Train only on old sticker crops; report agreement with held-out PnP pseudo labels."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from depth_data import FEATURE_NAMES, geometry_features, glyph_image, read_records, teacher_metrics
from sticker_depth_model import StickerDepthNet


class StickerDataset(Dataset):
    def __init__(self, root, rows, class_ids, augment=False):
        self.rows = rows
        self.images = [glyph_image(cv2.imread(str(root/row["crop"]))) for row in rows]
        self.geometry = np.stack([geometry_features(r["digit_bbox_xyxy"],r["camera_matrix"]) for r in rows])
        self.labels = [class_ids[r["digit"]] for r in rows]
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image = self.images[index].copy()
        if self.augment:
            image = np.clip(image*np.random.uniform(.75, 1.25)+np.random.uniform(-.025,.025), 0, 1)
        row = self.rows[index]
        weight = max(.2, 1/(1+row["teacher_rmse_px"]**2))
        return (torch.from_numpy(image), torch.from_numpy(self.geometry[index]),
                torch.tensor(row["depth_z_m"],dtype=torch.float32),
                torch.tensor(self.labels[index],dtype=torch.long), torch.tensor(weight,dtype=torch.float32))


def predict(model, loader):
    predictions, truths, class_outputs, labels = [], [], [], []
    model.eval()
    with torch.no_grad():
        for image, geometry, truth, cls, _ in loader:
            depth, logits = model(image, geometry)
            predictions.extend(depth.numpy().tolist())
            truths.extend(truth.numpy().tolist())
            class_outputs.extend(logits.argmax(1).numpy().tolist())
            labels.extend(cls.numpy().tolist())
    return np.asarray(truths), np.asarray(predictions), np.asarray(labels), np.asarray(class_outputs)


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    root, output = Path(args.dataset).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be empty")
    validation = json.loads((root/"validation_report.json").read_text())
    if validation["status"] != "passed":
        raise ValueError("Validate dataset before training")
    rows = read_records(root)
    split_rows = {split: [r for r in rows if r["split"] == split] for split in ("train","val","test")}
    sequence_splits = {}
    for row in rows:
        if row["sequence"] in sequence_splits and sequence_splits[row["sequence"]] != row["split"]:
            raise ValueError("A recording crosses train/val/test")
        sequence_splits[row["sequence"]] = row["split"]
    if min(len(r) for r in split_rows.values()) < 5:
        raise ValueError("Need at least five accepted examples in each recording-level split")
    class_values = sorted({r["digit"] for r in rows})
    class_ids = {value: i for i,value in enumerate(class_values)}
    trained_digits = sorted({r["digit"] for r in split_rows["train"]})
    if set(class_values) != set(trained_digits):
        raise ValueError("A held-out digit has no training examples")
    classification_enabled = len(trained_digits) > 1
    datasets = {s: StickerDataset(root, r, class_ids, augment=s=="train") for s,r in split_rows.items()}
    loaders = {s: DataLoader(d,batch_size=args.batch_size,shuffle=s=="train",num_workers=0) for s,d in datasets.items()}
    train_geometry = datasets["train"].geometry
    mean, std = train_geometry.mean(0), np.maximum(train_geometry.std(0), .05)
    train_depth = np.asarray([r["depth_z_m"] for r in split_rows["train"]])
    log_height = float(np.median(np.log(train_depth)+train_geometry[:,1]))
    median_depth = float(np.median(train_depth))
    model = StickerDepthNet(mean,std,log_height,len(class_values))
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    regression = nn.SmoothL1Loss(beta=.05,reduction="none")
    classification = nn.CrossEntropyLoss()
    output.mkdir(parents=True,exist_ok=True)
    config = dict(feature_names=FEATURE_NAMES,feature_mean=mean.tolist(),feature_std=std.tolist(),
        log_height_prior=log_height,train_depth_median_m=median_depth,class_values=class_values,
        classification_enabled=classification_enabled,training_digits=trained_digits,
        target="depth_z_m_camera_optical_axis",geometry_features="digit bbox and camera intrinsics only",
        crop_preprocessing="min(B,G,R), preserve aspect with 4px margin in 64x64",
        source="old_sticker_only",label_status="PnP pseudo labels; no independent range ground truth",
        roi_evaluation="teacher-guided glyph crops; end-to-end student digit detection not evaluated",
        training_dataset=str(root),seed=args.seed,torch_version=torch.__version__)
    (output/"model_config.json").write_text(json.dumps(config,indent=2)+"\n")
    best, best_epoch, history = float("inf"), -1, []
    for epoch in range(args.epochs):
        model.train(); total_loss=0.;samples=0
        for image, geometry, truth, cls, weight in loaders["train"]:
            depth, logits = model(image,geometry)
            loss = (regression(torch.log(depth),torch.log(truth))*weight).sum()/weight.sum()
            if classification_enabled:
                loss = loss+.1*classification(logits,cls)
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            total_loss += loss.item()*len(image); samples += len(image)
        truth, predicted, _, _ = predict(model,loaders["val"])
        metrics = teacher_metrics(truth,predicted)
        history.append(dict(epoch=epoch+1,train_loss=total_loss/samples,val=metrics))
        if metrics["mae_m"] < best:
            best,best_epoch=metrics["mae_m"],epoch+1
            torch.save(model.state_dict(),output/"best_weights.pt")
        if epoch==0 or (epoch+1)%5==0:
            print(f"epoch={epoch+1} loss={total_loss/samples:.5f} val_teacher_mae={metrics['mae_m']:.4f}m",flush=True)
        if epoch+1-best_epoch >= args.patience:
            break
    model.load_state_dict(torch.load(output/"best_weights.pt",map_location="cpu",weights_only=True))
    model.eval()
    report = dict(reference="held-out PnP pseudo labels, not independently measured distance",
        best_epoch=best_epoch,classification_enabled=classification_enabled,digit_counts=dict(Counter(r["digit"] for r in rows)),
        splits={},limitations=validation["limitations"]+["teacher-guided digit ROI at evaluation", "printed digit layout and camera are domain-specific"])
    prediction_file = (output/"predictions.jsonl").open("w")
    for split in ("train","val","test"):
        # Shuffling train data is disabled for report/provenance alignment.
        loader=DataLoader(StickerDataset(root,split_rows[split],class_ids),batch_size=args.batch_size,num_workers=0)
        truth,predicted,cls,out=predict(model,loader)
        geometry=datasets[split].geometry
        size_baseline=np.exp(log_height-geometry[:,1])
        constant=np.full(len(truth),median_depth)
        report["splits"][split]=dict(network=teacher_metrics(truth,predicted),
            inverse_digit_height=teacher_metrics(truth,size_baseline),constant_train_median=teacher_metrics(truth,constant),
            classification_accuracy=float(np.mean(cls==out)) if classification_enabled else None)
        for row,z in zip(split_rows[split],predicted):
            prediction_file.write(json.dumps(dict(id=row["id"],split=split,teacher_depth_z_m=row["depth_z_m"],
                predicted_depth_z_m=float(z),crop=row["crop"]))+"\n")
    prediction_file.close()
    val_metrics=report["splits"]["val"]
    improves=val_metrics["network"]["mae_m"] < val_metrics["inverse_digit_height"]["mae_m"]
    recommendation="network" if improves else "inverse_digit_height_baseline"
    config["network_beats_size_prior_on_validation"]=improves
    config["recommended_method"]=recommendation
    report["recommended_method_from_validation"]=recommendation
    (output/"model_config.json").write_text(json.dumps(config,indent=2)+"\n")
    example=(torch.zeros(1,1,64,64),torch.from_numpy(mean[None]))
    traced=torch.jit.trace(model,example)
    traced.save(str(output/"student.ts"))
    with torch.no_grad():
        eager=model(*example)[0];loaded=torch.jit.load(str(output/"student.ts"))(*example)[0]
        if not torch.allclose(eager,loaded):raise ValueError("Saved inference model disagrees with training model")
    (output/"history.json").write_text(json.dumps(history,indent=2)+"\n")
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    for name in ("train_sticker_depth.py","sticker_depth_model.py","depth_data.py","predict_sticker_depth.py",
                 "temporal_depth.py","depth_filter_config.json","STICKER_DEPTH.md"):
        shutil.copyfile(Path(__file__).with_name(name),output/name)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--output",required=True)
    parser.add_argument("--epochs",type=int,default=40)
    parser.add_argument("--patience",type=int,default=10)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--lr",type=float,default=.001)
    parser.add_argument("--seed",type=int,default=20261107)
    train(parser.parse_args())
