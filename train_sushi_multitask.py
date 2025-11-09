#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sushi Multitask Trainer
- Backbone: EfficientNet-B0 (torchvision)
- Heads: (A) class classification, (B) count classification (1/2/3貫)
- MixUp/CutMix: applied ONLY to class head; count head always uses clean images
- Supports: CSV mode (path,label_name) OR Folder mode (ImageFolder-like)
Author: ChatGPT
"""

import argparse
import csv
import os
import random
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ----------------------------
# Domain helpers (sushi / non-sushi handling)
# ----------------------------

def parse_keyword_list(s: str):
    return [w for w in [t.strip() for t in s.split(',')] if w]

DEFAULT_SUSHI_KWS = [
    "寿司","すし","鮨","にぎり","握り","軍艦","巻き","手巻き","巻","ちらし",
    "まぐろ","マグロ","中とろ","大とろ","トロ","サーモン","はまち","ハマチ","えび","海老",
    "いか","たこ","穴子","うに","イクラ","とびこ","たまご","玉子","炙り","ねぎとろ","とろたく",
    "貫"  # 明示的なヒント
]
DEFAULT_NON_SUSHI_KWS = [
    "ラーメン","らーめん","麺","うどん","そば","カレー","丼","唐揚げ","からあげ",
    "ポテト","フライドポテト","ナゲット","デザート","アイス","ケーキ","プリン","ゼリー",
    "ポップコーン"
]

def load_count_map(csv_path: str = "", json_path: str = ""):
    """
    Returns dict: label_name -> count (int in {0,1,2,3})
    Priority: JSON over CSV if both provided.
    """
    mapping = {}
    if json_path:
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    vi = int(v)
                except Exception:
                    continue
                if vi in (0,1,2,3):
                    mapping[str(k)] = vi
    elif csv_path:
        import csv
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not {'label_name','count'}.issubset(set(reader.fieldnames or [])):
                raise ValueError("count_map_csv must have columns: label_name,count")
            for row in reader:
                try:
                    vi = int(row['count'])
                except Exception:
                    continue
                if vi in (0,1,2,3):
                    mapping[str(row['label_name'])] = vi
    return mapping

def is_likely_sushi(label_name: str, sushi_kws=None, nonsushi_kws=None) -> bool:
    sushi_kws = sushi_kws or DEFAULT_SUSHI_KWS
    nonsushi_kws = nonsushi_kws or DEFAULT_NON_SUSHI_KWS
    # 非寿司キーワードが含まれていたら優先して非寿司扱い
    for kw in nonsushi_kws:
        if kw and kw in label_name:
            return False
    for kw in sushi_kws:
        if kw and kw in label_name:
            return True
    # どちらにも当てはまらなければ寿司としては扱わない（安全側）
    return False

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback


# ----------------------------
# Utils
# ----------------------------

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_count(label_name: str) -> int:
    """
    クラス名から貫数(1/2/3)を抽出して 0/1/2 に正規化（0=1貫,1=2貫,2=3貫）
    例: "まぐろ１貫まとめ", "三種盛り", "2貫", "二貫", "3貫", "三貫"
    """
    s = label_name

    # 全角数字を半角に寄せる簡易置換
    trans = str.maketrans("０１２３", "0123")
    s = s.translate(trans)

    # まず数字で検出（「種」も数の手掛かりとして扱う）
    m = re.search(r'(\d)\s*(貫|種)', s)
    if m:
        n = int(m.group(1))
        if n in (1, 2, 3):
            return n - 1

    # 漢数字パターン
    if re.search(r'(一|壱)\s*(貫|種)', s):
        return 0
    if re.search(r'(二|弐)\s*(貫|種)', s):
        return 1
    if re.search(r'(三|参)\s*(貫|種)', s):
        return 2

    # 「二種盛り/三種盛り」系
    if '二種' in s:
        return 1
    if '三種' in s:
        return 2

    # デフォルト（見つからない場合：1貫扱い）
    return 0


@dataclass
class Item:
    path: str
    label_name: str


def read_csv_items(csv_path: str) -> List[Item]:
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
                print(f"[WARN] File not found, skip: {p}")
    if len(items) == 0:
        raise ValueError(f"No valid items found in {csv_path}")
    return items


def scan_dir_items(root: str) -> List[Item]:
    """
    ImageFolder 風: root/CLASS_NAME/*.jpg という構造をスキャンして Item を作る
    クラス名はサブフォルダ名（label_nameにそのまま使う）
    """
    if not os.path.isdir(root):
        raise ValueError(f"Not a directory: {root}")

    items: List[Item] = []
    # サブフォルダを列挙（ソートして再現性確保）
    for cls in sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]):
        cls_dir = os.path.join(root, cls)
        for fn in os.listdir(cls_dir):
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                p = os.path.join(cls_dir, fn)
                items.append(Item(p, cls))
    if len(items) == 0:
        raise ValueError(f"No images found in {root}")
    return items


def build_class_mapping(items: List[Item]) -> Dict[str, int]:
    """
    CSV/Folder どちらでも使えるマッピング生成。
    ImageFolder と同じく「クラス名のソート順」で index を決める。
    """
    names = sorted(list({it.label_name for it in items}))
    return {n: i for i, n in enumerate(names)}


# ----------------------------
# Dataset & Transforms
# ----------------------------

class SushiDataset(Dataset):
    def __init__(self, items: List[Item], class_to_idx: Dict[str, int], transform=None, sushi_kws=None, nonsushi_kws=None, count_map=None):
        self.sushi_kws = sushi_kws
        self.nonsushi_kws = nonsushi_kws
        self.count_map = count_map or {}
        self.items = items
        self.class_to_idx = class_to_idx
        self.transform = transform
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        row = self.items[i]
        img = Image.open(row.path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y_cls = self.class_to_idx[row.label_name]
        # 1) explicit count map has highest priority
        if row.label_name in self.count_map:
            c = int(self.count_map[row.label_name])
            if c in (1,2,3):
                y_cnt = c - 1  # to 0..2
            else:  # 0 or others -> mask
                y_cnt = -1
        else:
            # 2) fallback to heuristic: only use when likely sushi AND label text contains count cues
            if is_likely_sushi(row.label_name, self.sushi_kws, self.nonsushi_kws):
                cnt = parse_count(row.label_name)
                if row.label_name is None or ("1" not in row.label_name and "１" not in row.label_name and
                                               "2" not in row.label_name and "２" not in row.label_name and
                                               "3" not in row.label_name and "３" not in row.label_name and
                                               ("一" not in row.label_name and "二" not in row.label_name and "三" not in row.label_name) and
                                               ("壱" not in row.label_name and "弐" not in row.label_name and "参" not in row.label_name) and
                                               ("二種" not in row.label_name and "三種" not in row.label_name)):
                    y_cnt = -1
                else:
                    y_cnt = cnt
            else:
                y_cnt = -1
        return img, y_cls, y_cnt


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


# ----------------------------
# Model
# ----------------------------

class EffNetB0_Count(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.class_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, n_classes)
        )
        self.count_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 3)  # 1/2/3 → 3 classes
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.class_head(feat), self.count_head(feat)


# ----------------------------
# MixUp / CutMix (class head only)
# ----------------------------

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
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)

    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))  # area-corrected
    return x_mix, y, y[idx], lam, 'cutmix'


def mix_criterion(logits, y_a, y_b, lam, label_smoothing=0.0):
    return lam * F.cross_entropy(logits, y_a, label_smoothing=label_smoothing) + \
           (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)


# ----------------------------
# Train / Eval
# ----------------------------

def train_one_epoch(model, loader, optimizer, scaler, device,
                    lam_count=0.5, mix_prob=0.7, label_smoothing=0.05):
    model.train()
    total_loss = total_cls = total_cnt = 0.0
    total_n = 0

    for x, y_cls, y_cnt in tqdm(loader, desc="train", leave=False):
        x = x.to(device); y_cls = y_cls.to(device); y_cnt = y_cnt.to(device)

        optimizer.zero_grad(set_to_none=True)

        # clean forward (used for count; also contributes to class loss)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits_cls_clean, logits_cnt = model(x)
            # Masked count loss
            mask = (y_cnt >= 0)
            if mask.any():
                loss_vec = F.cross_entropy(logits_cnt[mask], y_cnt[mask], weight=cnt_class_weights, reduction="mean")
                loss_cnt = loss_vec
            else:
                loss_cnt = torch.tensor(0.0, device=device)
            loss_cls_clean = F.cross_entropy(logits_cls_clean, y_cls, label_smoothing=label_smoothing)

        # mixed forward (class only)
        use_mix = (random.random() < mix_prob)
        loss_cls_mixed = torch.tensor(0.0, device=device)

        if use_mix:
            if random.random() < 0.5:
                x_mix, y_a, y_b, lam, _ = rand_mixup(x, y_cls, alpha=0.4)
            else:
                x_mix, y_a, y_b, lam, _ = rand_cutmix(x, y_cls, alpha=1.0)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits_cls_mixed, _ = model(x_mix)
                loss_cls_mixed = mix_criterion(logits_cls_mixed, y_a, y_b, lam, label_smoothing=label_smoothing)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            loss_cls = loss_cls_clean + loss_cls_mixed
            loss = loss_cls + lam_count * loss_cnt

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_cls += loss_cls.item() * bs
        total_cnt += loss_cnt.item() * bs
        total_n += bs

    return {
        "loss": total_loss / total_n,
        "loss_cls": total_cls / total_n,
        "loss_cnt": total_cnt / total_n,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct_cls = correct_cnt = total = 0
    cm_cnt = np.zeros((3, 3), dtype=int)
    cnt_total_eval = 0

    for x, y_cls, y_cnt in tqdm(loader, desc="valid", leave=False):
        x = x.to(device); y_cls = y_cls.to(device); y_cnt = y_cnt.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits_cls, logits_cnt = model(x)
        pred_cls = logits_cls.argmax(1)
        pred_cnt = logits_cnt.argmax(1)

        correct_cls += (pred_cls == y_cls).sum().item()
        total += x.size(0)

        # evaluate count only where label exists (y_cnt >= 0)
        mask = (y_cnt >= 0)
        if mask.any():
            correct_cnt += (pred_cnt[mask] == y_cnt[mask]).sum().item()
            cnt_total_eval += int(mask.sum().item())
            for t, p in zip(y_cnt[mask].cpu().numpy(), pred_cnt[mask].cpu().numpy()):
                cm_cnt[t, p] += 1

    acc_cnt = (correct_cnt / cnt_total_eval) if cnt_total_eval > 0 else 0.0
    return {
        "acc_cls": correct_cls / total,
        "acc_cnt": acc_cnt,
        "cm_cnt": cm_cnt,
        "cnt_eval_samples": cnt_total_eval,
    }


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Sushi Multitask Trainer (EffNet-B0)")
    # CSV mode
    parser.add_argument("--train_csv", type=str, help="CSV with columns: path,label_name")
    parser.add_argument("--val_csv", type=str, help="CSV with columns: path,label_name")
    # Folder mode
    parser.add_argument("--train_dir", type=str, help="ImageFolder-like root for training")
    parser.add_argument("--val_dir", type=str, help="ImageFolder-like root for validation")

    parser.add_argument("--img_size", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lam_count", type=float, default=0.5, help="Loss weight for count head")
    parser.add_argument("--mix_prob", type=float, default=0.7, help="Probability to apply MixUp/CutMix for class head")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--count_map_csv", type=str, default="", help="CSV with columns: label_name,count (0=non-sushi,1,2,3)")
    parser.add_argument("--count_map_json", type=str, default="", help="JSON object mapping label_name -> count (0/1/2/3)")
    parser.add_argument("--sushi_keywords", type=str, default="",
                        help="Comma-separated override keywords to detect sushi labels")
    parser.add_argument("--nonsushi_keywords", type=str, default="",
                        help="Comma-separated override keywords to detect non-sushi labels")
    args = parser.parse_args()

    if (args.train_dir and not args.val_dir) or (args.val_dir and not args.train_dir):
        raise ValueError("When using folder mode, please provide BOTH --train_dir and --val_dir")
    if (args.train_csv and not args.val_csv) or (args.val_csv and not args.train_csv):
        raise ValueError("When using CSV mode, please provide BOTH --train_csv and --val_csv")
    if not ((args.train_dir and args.val_dir) or (args.train_csv and args.val_csv)):
        raise ValueError("Provide either (train_dir & val_dir) OR (train_csv & val_csv)")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # ---------- Load items ----------
    print("==> Loading items...")
    if args.train_dir and args.val_dir:
        train_items = scan_dir_items(args.train_dir)
        val_items = scan_dir_items(args.val_dir)
        mode = "folder"
    else:
        train_items = read_csv_items(args.train_csv)
        val_items = read_csv_items(args.val_csv)
        mode = "csv"

    # Build label mapping from BOTH train & val (safer, ImageFolderと同じくクラス名ソート)
    class_to_idx = build_class_mapping(train_items + val_items)
    n_classes = len(class_to_idx)
    print(f"Mode: {mode} | Classes: {n_classes}")
    print("Example classes (first 10):", list(class_to_idx.keys())[:10])

    # ---------- Datasets ----------
    train_tf, val_tf = build_transforms(args.img_size)
    sushi_kws = parse_keyword_list(args.sushi_keywords) if args.sushi_keywords else DEFAULT_SUSHI_KWS
    nonsushi_kws = parse_keyword_list(args.nonsushi_keywords) if args.nonsushi_keywords else DEFAULT_NON_SUSHI_KWS

    count_map = load_count_map(csv_path=args.count_map_csv, json_path=args.count_map_json)
    if count_map:
        print(f"Loaded count map entries: {len(count_map)}")

    train_ds = SushiDataset(train_items, class_to_idx, transform=train_tf,
                            sushi_kws=sushi_kws, nonsushi_kws=nonsushi_kws, count_map=count_map)
    val_ds   = SushiDataset(val_items,   class_to_idx, transform=val_tf,
                            sushi_kws=sushi_kws, nonsushi_kws=nonsushi_kws, count_map=count_map)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # ---------- Model / Opt ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EffNetB0_Count(n_classes=n_classes).to(device)

    # ----- Build count class weights (only from sushi & available count labels) -----
    cnt_hist = np.zeros(3, dtype=np.int64)
    for _, _, y_cnt in DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=0):
        y_cnt = np.array(y_cnt)
        for k in (0,1,2):
            cnt_hist[k] += int((y_cnt == k).sum())
    # Avoid zero division; inverse frequency weighting
    cnt_weights = np.array([1.0/(h if h>0 else 1.0) for h in cnt_hist], dtype=np.float32)
    cnt_weights = cnt_weights / cnt_weights.sum() * 3.0  # normalize to mean 1
    cnt_class_weights = torch.tensor(cnt_weights, dtype=torch.float32, device=device)
    print("Count label histogram (train, mask excl.):", cnt_hist.tolist())
    print("Count class weights:", cnt_weights.tolist())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc = 0.0
    best_path = os.path.join(args.out_dir, "best.pt")

    print("==> Start training")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device,
                             lam_count=args.lam_count, mix_prob=args.mix_prob, label_smoothing=0.05)
        ev = evaluate(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        print(f"[{epoch:02d}/{args.epochs}] {dt:.1f}s | "
              f"loss={tr['loss']:.4f} (cls {tr['loss_cls']:.4f}/cnt {tr['loss_cnt']:.4f}) | "
              f"acc_cls={ev['acc_cls']:.4f} acc_cnt={ev['acc_cnt']:.4f} (n={ev.get('cnt_eval_samples',0)})")
        print("count confusion matrix (rows=true 1/2/3, cols=pred):")
        print(ev['cm_cnt'])

        # Save best by class accuracy primarily; tie-breaker by count accuracy
        score = ev["acc_cls"] + 0.2 * ev["acc_cnt"]
        if score > best_acc:
            best_acc = score
            torch.save({
                "model": model.state_dict(),
                "class_to_idx": class_to_idx,
                "args": vars(args),
                "epoch": epoch,
                "metrics": ev,
            }, best_path)
            print(f"==> Saved best to {best_path} (score {best_acc:.4f})")

    # Save final
    final_path = os.path.join(args.out_dir, "final.pt")
    torch.save({
        "model": model.state_dict(),
        "class_to_idx": class_to_idx,
        "args": vars(args),
        "epoch": args.epochs,
    }, final_path)
    print(f"==> Saved final to {final_path}")
    print("Done.")


if __name__ == "__main__":
    main()
