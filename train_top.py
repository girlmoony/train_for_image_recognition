# sushi_top_model.py
# 真上から撮影したお寿司画像用モデル（EfficientNet-B0 fine-tuning）
# - ImageNet 事前学習
# - 真上専用の data augmentation
# - classifier だけ Dropout p=0.4
# - CrossEntropy + label_smoothing
# - AdamW + 異なる学習率（backbone / classifier）
# - ReduceLROnPlateau スケジューラ

import os
import cv2
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


# =========================
#  Config
# =========================
IMG_SIZE = 256
NUM_CLASSES = 250
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_FREQ = 10  # 学習中のログ出力頻度（step単位）


# =========================
#  Dataset
# =========================
class SushiDataset(Dataset):
    """
    画像パスとラベルのリストから読み込むシンプルな Dataset
    - 画像読み込み：cv2.imread
    - BGR → RGB に変換
    - transforms に渡して tensor へ
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
    ):
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(image_path)  # BGR, uint8
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        label = int(label)
        return img, label


# =========================
#  Transforms（真上専用）
# =========================
def get_transforms(img_size: int = IMG_SIZE):
    # train: 真上カメラの揺らぎを強めるような augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),

        # 位置ズレ・ズーム・わずかな縦横比の違いを再現
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
        ),

        # 明るさ・色の揺らぎ（照明やカメラ差を再現）
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        ),

        # 真上なら180°回転してもほぼ同じ “寿司”
        transforms.RandomRotation(degrees=180),
        transforms.RandomHorizontalFlip(p=0.5),

        # ピントの甘さ・少しのブレを再現
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))],
            p=0.3,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),

        # 一部が隠れる / 影になる状況の再現
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.06),
            ratio=(0.3, 3.3),
            value=0,
            inplace=False,
        ),
    ])

    # val / test: ランダム無し。サイズを合わせるのみ
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform


# =========================
#  Model（EfficientNet-B0）
# =========================
def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    EfficientNet-B0 (ImageNet pretrained) をベースに、
    classifier を置き換えて Dropout p=0.4 + Linear(num_classes) にする
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )

    return model


# =========================
#  Optimizer & Scheduler
# =========================
def build_optimizer_and_scheduler(model: nn.Module):
    """
    AdamW + パラメータグループ
    - backbone: lr = 1e-4
    - classifier: lr = 1e-3
    Scheduler: ReduceLROnPlateau（val_loss 監視）
    """
    features_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if "classifier" in name:
            classifier_params.append(param)
        else:
            features_params.append(param)

    optimizer = AdamW(
        [
            {"params": features_params,   "lr": 1e-4},
            {"params": classifier_params, "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True,
    )

    return optimizer, scheduler


# =========================
#  Train / Eval Loop
# =========================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device = DEVICE,
):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for step, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)
        running_loss += loss.item() * labels.size(0)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)

        if (step + 1) % PRINT_FREQ == 0:
            avg_loss = running_loss / running_total
            avg_acc = running_correct / running_total
            print(
                f"[Epoch {epoch}][Step {step+1}/{len(loader)}] "
                f"loss={avg_loss:.4f}, acc={avg_acc:.4f}"
            )

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = outputs.max(1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


# =========================
#  Main
# =========================
def main():
    # --- ここはあなたの環境に合わせて埋めてください ---
    # 例：
    # train_image_paths, train_labels = load_from_csv("train.csv")
    # val_image_paths,   val_labels   = load_from_csv("val.csv")

    train_image_paths: List[str] = []  # 画像パスのリスト（train）
    train_labels: List[int] = []       # ラベル（0〜NUM_CLASSES-1）
    val_image_paths: List[str] = []    # 画像パスのリスト（val）
    val_labels: List[int] = []         # ラベル（0〜NUM_CLASSES-1）

    if len(train_image_paths) == 0:
        raise RuntimeError("train_image_paths が空です。パスとラベルを設定してください。")

    # transforms
    train_transform, val_transform = get_transforms(IMG_SIZE)

    # datasets / loaders
    train_dataset = SushiDataset(
        image_paths=train_image_paths,
        labels=train_labels,
        transform=train_transform,
    )
    val_dataset = SushiDataset(
        image_paths=val_image_paths,
        labels=val_labels,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # model / optimizer / scheduler / loss
    model = build_model(NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer, scheduler = build_optimizer_and_scheduler(model)

    best_val_acc = 0.0
    best_model_path = "best_sushi_top_model.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, DEVICE
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE
        )

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # ReduceLROnPlateau 用
        scheduler.step(val_loss)

        # ベストモデル保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> best model updated: val_acc={best_val_acc:.4f}")

    print(f"\nTraining finished. Best val_acc = {best_val_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
