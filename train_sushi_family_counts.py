#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multitask Trainer: Class (330) + Family (nigiri/gunkan/tempura/other) + Family-specific Count Heads
- Shared backbone (EfficientNet-B0, torchvision)
- Heads:
  * Class head: 330-way (or N classes from your dataset)
  * Family head: 4-way (nigiri, gunkan, tempura, other)
  * Count heads (3-way each): count_nigiri, count_gunkan, count_tempura
- Loss:
  L = L_class + α * L_family + λ_n * L_cnt_n + λ_g * L_cnt_g + λ_t * L_cnt_t
  where each L_cnt_* is computed ONLY on samples of that family with a valid count label.
- MixUp/CutMix are applied to the CLASS head only. Count heads always train on clean images.
- Inputs:
  * Either folder mode (ImageFolder-like) or CSV mode (path,label_name)
  * count_map_csv: label_name,count (0=non-sushi/ignore, 1/2/3=貫数)   [REQUIRED]
  * family_map_csv: label_name,family where family∈{nigiri,gunkan,tempura,other}  [REQUIRED]
"""

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
FAMILIES = ['nigiri', 'gunkan', 'tempura', 'other']
F2I = {f: i for i, f in enumerate(FAMILIES)}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class Item:
    path: str
    label_name: str


def read_items_from_csv(csv_path: str) -> List[Item]:
    items: List[Item] = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not {'path', 'label_name'}.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must have columns: path,label_name")
        for row in reader:
            p = row['path']
            n = row['label_name']
            if os.path.isfile(p):
                items.append(Item(p, n))
            else:
                print(f"[WARN] file missing, skip: {p}")
    if not items:
        raise ValueError(f"No valid items in {csv_path}")
    return items


def read_items_from_folder(root: str) -> List[Item]:
    if not os.path.isdir(root):
        raise ValueError(f"Not a directory: {root}")
    items: List[Item] = []
    for cls in sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]):
        cls_dir = os.path.join(root, cls)
        for fn in os.listdir(cls_dir):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                items.append(Item(os.path.join(cls_dir, fn), cls))
    if not items:
        raise ValueError(f"No images found in {root}")
    return items


def build_class_mapping(items: List[Item]) -> Dict[str, int]:
    names = sorted(list({it.label_name for it in items}))
    return {n: i for i, n in enumerate(names)}


def load_count_map_csv(csv_path: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not {'label_name', 'count'}.issubset(set(reader.fieldnames or [])):
            raise ValueError("count_map_csv must have columns: label_name,count")
        for row in reader:
            try:
                c = int(row['count'])
            except Exception:
                continue
            if c in (0, 1, 2, 3):
                mapping[str(row['label_name'])] = c
    if not mapping:
        raise ValueError("count_map_csv is empty or invalid")
    return mapping


def load_family_map_csv(csv_path: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not {'label_name', 'family'}.issubset(set(reader.fieldnames or [])):
            raise ValueError("family_map_csv must have columns: label_name,family")
        for row in reader:
            fam = str(row['family']).strip().lower()
            if fam not in F2I:
                raise ValueError(f"Invalid family '{fam}' for label '{row['label_name']}'. Allowed: {list(F2I.keys())}")
            mapping[str(row['label_name'])] = F2I[fam]
    if not mapping:
        raise ValueError("family_map_csv is empty or invalid")
    return mapping


class SushiFamilyDataset(Dataset):
    def __init__(self, items: List[Item], class_to_idx: Dict[str, int],
                 count_map: Dict[str, int], family_map: Dict[str, int],
                 transform=None):
        self.items = items
        self.class_to_idx = class_to_idx
        self.count_map = count_map
        self.family_map = family_map
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it.path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        y_class = self.class_to_idx[it.label_name]

        if it.label_name not in self.family_map:
            raise KeyError(f"family_map missing label_name: {it.label_name}")
        y_family = self.family_map[it.label_name]  # 0..3

        if it.label_name not in self.count_map:
            raise KeyError(f"count_map missing label_name: {it.label_name}")
        c = int(self.count_map[it.label_name])
        y_count = (c - 1) if c in (1, 2, 3) else -1  # -1 => masked

        return img, y_class, y_family, y_count


def build_transforms(img_size: int = 300):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.05, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.15),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    return train_tf, val_tf


class EffNetB0_MTFamily(nn.Module):
    def __init__(self, n_classes: int, n_families: int = 4):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.class_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, n_classes))
        self.family_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, n_families))

        self.count_head_nigiri  = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 3))
        self.count_head_gunkan  = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 3))
        self.count_head_tempura = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 3))

    def forward(self, x):
        feat = self.backbone(x)
        logits_class  = self.class_head(feat)
        logits_family = self.family_head(feat)
        logits_cnt_n  = self.count_head_nigiri(feat)
        logits_cnt_g  = self.count_head_gunkan(feat)
        logits_cnt_t  = self.count_head_tempura(feat)
        return logits_class, logits_family, logits_cnt_n, logits_cnt_g, logits_cnt_t


def rand_mixup(x, y, alpha=0.4):
    if alpha is None or alpha <= 0:
        return x, y, None, 1.0, 'none'
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam, 'mixup'


def rand_cutmix(x, y, alpha=1.0):
    if alpha is None or alpha <= 0:
        return x, y, None, 1.0, 'none'
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    B, C, H, W = x.size()
    idx = torch.randperm(B, device=x.device)
    cut_w = int(W * (1 - lam) ** 0.5)
    cut_h = int(H * (1 - lam) ** 0.5)
    cx = random.randint(0, W); cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)
    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x_mix, y, y[idx], lam, 'cutmix'


def mix_criterion(logits, y_a, y_b, lam, label_smoothing=0.0):
    return lam * F.cross_entropy(logits, y_a, label_smoothing=label_smoothing) + \
           (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)


def train_one_epoch(model, loader, optimizer, scaler, device,
                    alpha_family=0.3,
                    lambda_n=0.6, lambda_g=0.6, lambda_t=0.3,
                    mix_prob=0.7, label_smoothing=0.05):
    model.train()
    tot_loss = tot_cls = tot_fam = tot_cnt_n = tot_cnt_g = tot_cnt_t = 0.0
    N = 0

    for x, y_cls, y_fam, y_cnt in tqdm(loader, desc="train", leave=False):
        x = x.to(device); y_cls = y_cls.to(device); y_fam = y_fam.to(device); y_cnt = y_cnt.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits_cls, logits_fam, log_n, log_g, log_t = model(x)

            L_cls = F.cross_entropy(logits_cls, y_cls, label_smoothing=label_smoothing)
            L_fam = F.cross_entropy(logits_fam, y_fam)

            m_n = (y_fam == F2I['nigiri']) & (y_cnt >= 0)
            m_g = (y_fam == F2I['gunkan']) & (y_cnt >= 0)
            m_t = (y_fam == F2I['tempura']) & (y_cnt >= 0)

            L_cnt_n = F.cross_entropy(log_n[m_n], y_cnt[m_n]) if m_n.any() else torch.tensor(0.0, device=device)
            L_cnt_g = F.cross_entropy(log_g[m_g], y_cnt[m_g]) if m_g.any() else torch.tensor(0.0, device=device)
            L_cnt_t = F.cross_entropy(log_t[m_t], y_cnt[m_t]) if m_t.any() else torch.tensor(0.0, device=device)

        L_cls_mix = torch.tensor(0.0, device=device)
        if random.random() < mix_prob:
            if random.random() < 0.5:
                x_mix, y_a, y_b, lam, _ = rand_mixup(x, y_cls, alpha=0.4)
            else:
                x_mix, y_a, y_b, lam, _ = rand_cutmix(x, y_cls, alpha=1.0)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits_cls_mix, _, _, _, _ = model(x_mix)
                L_cls_mix = mix_criterion(logits_cls_mix, y_a, y_b, lam, label_smoothing=label_smoothing)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            L_cnt = lambda_n * L_cnt_n + lambda_g * L_cnt_g + lambda_t * L_cnt_t
            loss = L_cls + L_cls_mix + alpha_family * L_fam + L_cnt

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        tot_loss += loss.item() * bs
        tot_cls  += (L_cls + L_cls_mix).item() * bs
        tot_fam  += L_fam.item() * bs
        tot_cnt_n += L_cnt_n.item() * bs
        tot_cnt_g += L_cnt_g.item() * bs
        tot_cnt_t += L_cnt_t.item() * bs
        N += bs

    return {
        "loss": tot_loss / N,
        "loss_cls": tot_cls / N,
        "loss_family": tot_fam / N,
        "loss_cnt_n": tot_cnt_n / N,
        "loss_cnt_g": tot_cnt_g / N,
        "loss_cnt_t": tot_cnt_t / N,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct_cls = 0
    correct_fam = 0

    fam_keys = ['nigiri', 'gunkan', 'tempura']
    cnt_correct = {k: 0 for k in fam_keys}
    cnt_total   = {k: 0 for k in fam_keys}
    cm = {k: np.zeros((3, 3), dtype=int) for k in fam_keys}

    for x, y_cls, y_fam, y_cnt in tqdm(loader, desc="valid", leave=False):
        x = x.to(device); y_cls = y_cls.to(device); y_fam = y_fam.to(device); y_cnt = y_cnt.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            log_cls, log_fam, log_n, log_g, log_t = model(x)

        pred_cls = log_cls.argmax(1)
        pred_fam = log_fam.argmax(1)
        correct_cls += (pred_cls == y_cls).sum().item()
        correct_fam += (pred_fam == y_fam).sum().item()
        total += x.size(0)

        for fam_name in fam_keys:
            fam_id = F2I[fam_name]
            mask = (y_fam == fam_id) & (y_cnt >= 0)
            if not mask.any():
                continue
            if fam_name == 'nigiri':
                pred_cnt = log_n.argmax(1)
            elif fam_name == 'gunkan':
                pred_cnt = log_g.argmax(1)
            else:
                pred_cnt = log_t.argmax(1)

            cnt_correct[fam_name] += (pred_cnt[mask] == y_cnt[mask]).sum().item()
            cnt_total[fam_name]   += int(mask.sum().item())

            y_true = y_cnt[mask].detach().cpu().numpy()
            y_pred = pred_cnt[mask].detach().cpu().numpy()
            for t, p in zip(y_true, y_pred):
                cm[fam_name][t, p] += 1

    results = {
        "acc_class": correct_cls / total,
        "acc_family": correct_fam / total,
        "acc_count_overall": (sum(cnt_correct.values()) / max(1, sum(cnt_total.values()))),
        "acc_count_by_family": {k: (cnt_correct[k] / cnt_total[k]) if cnt_total[k] > 0 else 0.0 for k in fam_keys},
        "count_samples_by_family": cnt_total,
        "cm_by_family": cm,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Multitask: class + family + family-specific counts")
    parser.add_argument("--train_csv", type=str, help="CSV with columns: path,label_name")
    parser.add_argument("--val_csv", type=str, help="CSV with columns: path,label_name")
    parser.add_argument("--train_dir", type=str, help="ImageFolder-like train root")
    parser.add_argument("--val_dir", type=str, help="ImageFolder-like val root")

    parser.add_argument("--count_map_csv", type=str, required=True, help="CSV: label_name,count (0/1/2/3)")
    parser.add_argument("--family_map_csv", type=str, required=True, help="CSV: label_name,family (nigiri/gunkan/tempura/other)")

    parser.add_argument("--img_size", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--alpha_family", type=float, default=0.3)
    parser.add_argument("--lambda_n", type=float, default=0.6)
    parser.add_argument("--lambda_g", type=float, default=0.6)
    parser.add_argument("--lambda_t", type=float, default=0.3)

    parser.add_argument("--mix_prob", type=float, default=0.7)
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if (args.train_dir and not args.val_dir) or (args.val_dir and not args.train_dir):
        raise ValueError("Provide BOTH --train_dir and --val_dir for folder mode")
    if (args.train_csv and not args.val_csv) or (args.val_csv and not args.train_csv):
        raise ValueError("Provide BOTH --train_csv and --val_csv for CSV mode")
    if not ((args.train_dir and args.val_dir) or (args.train_csv and args.val_csv)):
        raise ValueError("Provide either (train_dir & val_dir) OR (train_csv & val_csv)")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    print("==> Loading items...")
    if args.train_dir:
        train_items = read_items_from_folder(args.train_dir)
        val_items   = read_items_from_folder(args.val_dir)
        mode = "folder"
    else:
        train_items = read_items_from_csv(args.train_csv)
        val_items   = read_items_from_csv(args.val_csv)
        mode = "csv"

    class_to_idx = build_class_mapping(train_items + val_items)
    n_classes = len(class_to_idx)
    print(f"Mode: {mode} | Classes: {n_classes}")

    count_map  = load_count_map_csv(args.count_map_csv)
    family_map = load_family_map_csv(args.family_map_csv)
    print(f"Loaded count map entries: {len(count_map)} | family map entries: {len(family_map)}")

    train_tf, val_tf = build_transforms(args.img_size)
    train_ds = SushiFamilyDataset(train_items, class_to_idx, count_map, family_map, transform=train_tf)
    val_ds   = SushiFamilyDataset(val_items,   class_to_idx, count_map, family_map, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EffNetB0_MTFamily(n_classes=n_classes, n_families=len(FAMILIES)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_score = -1e9
    best_path = os.path.join(args.out_dir, "best.pt")

    print("==> Start training")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device,
                             alpha_family=args.alpha_family,
                             lambda_n=args.lambda_n, lambda_g=args.lambda_g, lambda_t=args.lambda_t,
                             mix_prob=args.mix_prob, label_smoothing=0.05)
        ev = evaluate(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        print(f"[{epoch:02d}/{args.epochs}] {dt:.1f}s | "
              f"loss={tr['loss']:.4f} | "
              f"class={tr['loss_cls']:.4f} fam={tr['loss_family']:.4f} "
              f"cnt(n/g/t)={tr['loss_cnt_n']:.4f}/{tr['loss_cnt_g']:.4f}/{tr['loss_cnt_t']:.4f}")
        print(f"  acc_class={ev['acc_class']:.4f} acc_family={ev['acc_family']:.4f} "
              f"acc_count_overall={ev['acc_count_overall']:.4f} "
              f"acc_count_by_family={ev['acc_count_by_family']} "
              f"count_samples_by_family={ev['count_samples_by_family']}")

        score = ev["acc_class"] + 0.2 * ev["acc_family"] + 0.2 * ev["acc_count_overall"]
        if score > best_score:
            best_score = score
            torch.save({
                "model": model.state_dict(),
                "class_to_idx": class_to_idx,
                "family_index": F2I,
                "args": vars(args),
                "epoch": epoch,
                "metrics": ev,
            }, best_path)
            print(f"==> Saved best to {best_path} (score {best_score:.4f})")

        for fam in ['nigiri','gunkan','tempura']:
            cm = ev['cm_by_family'][fam]
            if cm.sum() > 0:
                print(f"  CM[{fam}] rows=true(0..2) cols=pred:\n{cm}")

    final_path = os.path.join(args.out_dir, "final.pt")
    torch.save({
        "model": model.state_dict(),
        "class_to_idx": class_to_idx,
        "family_index": F2I,
        "args": vars(args),
        "epoch": args.epochs,
    }, final_path)
    print(f"==> Saved final to {final_path}")
    print("Done.")


if __name__ == "__main__":
    main()

python train_sushi_family_counts.py \
  --train_dir /path/to/train_dir \
  --val_dir   /path/to/val_dir \
  --count_map_csv  /path/to/count_map.csv \
  --family_map_csv /path/to/family_map.csv \
  --img_size 300 --batch_size 64 --epochs 20 --lr 3e-4 --out_dir ./outputs
