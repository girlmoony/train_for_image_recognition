import re

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import models


import random
import torch
import torch.nn.functional as F

def parse_count(label_name: str) -> int:
    """
    クラス名から貫数(1/2/3)を抽出して 0/1/2 に正規化（0=1貫,1=2貫,2=3貫）
    例: "まぐろ１貫まとめ", "三種盛り", "2貫", "二貫", "3貫", "三貫"
    """
    s = label_name

    # 全角数字を半角に寄せる簡易置換
    trans = str.maketrans("０１２３", "0123")
    s = s.translate(trans)

    # まず数字で検出
    m = re.search(r'(\d)\s*(貫|種)', s)
    if m:
        n = int(m.group(1))
        if n in (1,2,3): return n - 1

    # 漢数字パターン
    if re.search(r'(一|壱)\s*(貫|種)', s): return 0
    if re.search(r'(二|弐)\s*(貫|種)', s): return 1
    if re.search(r'(三|参)\s*(貫|種)', s): return 2

    # 「二種盛り/三種盛り」系
    if '二種' in s: return 1
    if '三種' in s: return 2

    # デフォルト（見つからない場合は1貫扱い or 例外にする）
    return 0


# 推奨Augment（数が欠けない範囲で弱め）
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(300, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomRotation(degrees=20),             # 180°→±20°
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.05, hue=0.02),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.15),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize(330),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
])

class SushiDataset(Dataset):
    def __init__(self, items, class_to_idx, transform=None):
        """
        items: List[{"path":..., "label_name":...}]
        class_to_idx: {label_name: class_id}
        """
        self.items = items
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        row = self.items[i]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform: img = self.transform(img)

        cls_id = self.class_to_idx[row["label_name"]]
        cnt_id = parse_count(row["label_name"])  # 0/1/2
        return img, cls_id, cnt_id



class EffNetB0_Count(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  # グローバルプーリング後の特徴を出す

        self.class_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, n_classes)
        )
        self.count_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 3)  # 1/2/3 → 3クラス
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.class_head(feat), self.count_head(feat)


def rand_mixup(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, None, 1.0, 'none'
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam, 'mixup'

def rand_cutmix(x, y, alpha=1.0):
    if alpha <= 0:
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
    # 実面積でlamを補正
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x_mix, y, y[idx], lam, 'cutmix'

def mix_criterion(logits, y_a, y_b, lam):
    return lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, scaler, device, lam_count=0.5, mix_prob=0.7):
    model.train()
    total_loss = total_cls = total_cnt = 0.0

    for x, y_cls, y_cnt in loader:
        x = x.to(device); y_cls = y_cls.to(device); y_cnt = y_cnt.to(device)

        optimizer.zero_grad(set_to_none=True)

        # ---- clean forward（count用 + classにも使って良い）----
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits_cls_clean, logits_cnt = model(x)
            loss_cnt = F.cross_entropy(logits_cnt, y_cnt)

            # クラス損失（クリーン分）— 任意：0.5倍などで寄与させても良い
            loss_cls_clean = F.cross_entropy(logits_cls_clean, y_cls)

        # ---- mixed forward（class用のみ）----
        use_mix = (random.random() < mix_prob)
        loss_cls_mixed = torch.tensor(0.0, device=device)

        if use_mix:
            # MixUp or CutMix をランダムで
            if random.random() < 0.5:
                x_mix, y_a, y_b, lam, kind = rand_mixup(x, y_cls, alpha=0.4)
            else:
                x_mix, y_a, y_b, lam, kind = rand_cutmix(x, y_cls, alpha=1.0)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits_cls_mixed, _ = model(x_mix)  # countヘッドは使わない
                loss_cls_mixed = mix_criterion(logits_cls_mixed, y_a, y_b, lam)

        # ---- 総損失 ----
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            loss_cls = loss_cls_clean + loss_cls_mixed
            loss = loss_cls + lam_count * loss_cnt

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        total_cls  += loss_cls.item() * x.size(0)
        total_cnt  += loss_cnt.item() * x.size(0)

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "loss_cls": total_cls / n,
        "loss_cnt": total_cnt / n,
    }
import numpy as np

def evaluate(model, loader, device):
    model.eval()
    correct_cls = correct_cnt = total = 0
    cm_cnt = np.zeros((3,3), dtype=int)

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):
        for x, y_cls, y_cnt in loader:
            x = x.to(device); y_cls = y_cls.to(device); y_cnt = y_cnt.to(device)
            logits_cls, logits_cnt = model(x)
            pred_cls = logits_cls.argmax(1)
            pred_cnt = logits_cnt.argmax(1)

            correct_cls += (pred_cls == y_cls).sum().item()
            correct_cnt += (pred_cnt == y_cnt).sum().item()
            total += x.size(0)

            for t, p in zip(y_cnt.cpu().numpy(), pred_cnt.cpu().numpy()):
                cm_cnt[t, p] += 1

    return {
        "acc_cls": correct_cls / total,
        "acc_cnt": correct_cnt / total,
        "cm_cnt": cm_cnt,
    }
import torch
from torch.cuda.amp import GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 準備：items, class_to_idx を作成しておく
train_ds = SushiDataset(train_items, class_to_idx, transform=train_tf)
val_ds   = SushiDataset(val_items,   class_to_idx, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

model = EffNetB0_Count(n_classes=len(class_to_idx)).to(device)
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=20)  # 例: 20エポック
scaler = GradScaler()

for epoch in range(20):
    tr = train_one_epoch(model, train_loader, optimizer, scaler, device, lam_count=0.5, mix_prob=0.7)
    ev = evaluate(model, val_loader, device)
    scheduler.step()

    print(f"[{epoch:02d}] loss={tr['loss']:.4f} (cls {tr['loss_cls']:.4f}/cnt {tr['loss_cnt']:.4f}) | "
          f"acc_cls={ev['acc_cls']:.4f} acc_cnt={ev['acc_cnt']:.4f}")
    print("count confusion matrix (rows=true 1/2/3, cols=pred):")
    print(ev["cm_cnt"])



