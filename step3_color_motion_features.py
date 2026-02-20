"""Step 3: Color & Motion Feature Fusion — Experiment 2
-------------------------------------------------------
Extends the ResNet baseline by appending hand-crafted color and texture
descriptors to the CNN feature vector before the classification head.

Feature pipeline
----------------
    Image → ResNet backbone (frozen) → 2048-d embedding
                                          ↓  concat
    Image → HSV color histogram       → 96-d  vector
    Image → LBP texture histogram     → 256-d vector
                                          ↓
                       Fusion MLP (dropout → 512 → 6)

The backbone stays *frozen* so the experiment isolates the contribution of
the hand-crafted features; the only trainable weights are the fusion MLP.

Usage
-----
    python step3_color_motion_features.py

    # Custom settings
    python step3_color_motion_features.py \\
        --data_dir mood_dataset \\
        --backbone_ckpt checkpoints/best_model.pth \\
        --epochs 20 \\
        --lr 1e-3 \\
        --batch_size 32

Outputs
-------
    checkpoints/best_model_exp2.pth
    results/exp2_training_curves.png
    results/exp2_confusion_matrix.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOOD_LABELS: list[str] = [
    "calm",
    "cheerful",
    "energetic",
    "dark_tense",
    "dramatic",
    "melancholic",
]
NUM_CLASSES = len(MOOD_LABELS)

# Color histogram bins per channel (H, S, V)
HSV_BINS = (32, 32, 32)
COLOR_DIM = sum(HSV_BINS)  # 96

# Local Binary Pattern settings
LBP_RADIUS = 2
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_N_BINS = LBP_N_POINTS + 3  # uniform LBP bin count: n_points + 3
LBP_REPEATS = 4  # spatial grid (2×2)
TEXTURE_DIM = LBP_N_BINS * LBP_REPEATS  # 256

# ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Hand-crafted feature extractors
# ---------------------------------------------------------------------------


def extract_hsv_histogram(img_np: np.ndarray) -> np.ndarray:
    """Compute a normalised HSV colour histogram from an RGB numpy image.

    Args:
        img_np: ``uint8`` array of shape ``(H, W, 3)``, RGB channel order.

    Returns:
        Flat float32 vector of length ``sum(HSV_BINS)`` (96).
    """
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    features: list[np.ndarray] = []
    for c, bins in enumerate(HSV_BINS):
        hist = cv2.calcHist([hsv], [c], None, [bins], [0, 256])
        hist = hist.flatten().astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        features.append(hist)
    return np.concatenate(features)


def _lbp_histogram(gray_patch: np.ndarray) -> np.ndarray:
    """Compute a uniform LBP histogram for a single patch.

    Args:
        gray_patch: Greyscale ``uint8`` array.

    Returns:
        Normalised float32 histogram of length ``LBP_N_BINS``.
    """
    # Manually compute LBP to avoid scikit-image dependency
    r = LBP_RADIUS
    h, w = gray_patch.shape
    lbp = np.zeros((h - 2 * r, w - 2 * r), dtype=np.uint8)
    center = gray_patch[r:-r, r:-r].astype(np.int16)
    angles = np.linspace(0, 2 * np.pi, LBP_N_POINTS, endpoint=False)
    code = np.zeros_like(center, dtype=np.uint8)
    for bit, angle in enumerate(angles):
        y = round(r * np.sin(angle))
        x = round(r * np.cos(angle))
        neighbor = gray_patch[r + y : h - r + y, r + x : w - r + x].astype(np.int16)
        code |= ((neighbor >= center).astype(np.uint8)) << bit

    hist, _ = np.histogram(code.flatten(), bins=LBP_N_BINS, range=(0, LBP_N_BINS))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def extract_lbp_texture(img_np: np.ndarray) -> np.ndarray:
    """Compute a spatial-grid LBP texture histogram.

    Divides the image into a 2×2 grid and concatenates a LBP histogram from
    each cell, yielding a spatially-aware texture descriptor.

    Args:
        img_np: ``uint8`` array of shape ``(H, W, 3)``, RGB channel order.

    Returns:
        Flat float32 vector of length ``TEXTURE_DIM`` (256).
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    grid = 2  # 2×2 spatial grid
    ph, pw = h // grid, w // grid
    histograms: list[np.ndarray] = []
    for r in range(grid):
        for c in range(grid):
            patch = gray[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
            histograms.append(_lbp_histogram(patch))
    return np.concatenate(histograms)


# ---------------------------------------------------------------------------
# Dataset wrapper — pairs PIL images with pre-computed hand-crafted features
# ---------------------------------------------------------------------------

_cnn_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

_aug_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class MoodDatasetWithFeatures(Dataset):
    """ImageFolder dataset augmented with colour + texture feature vectors.

    Args:
        root: Directory with one subdirectory per mood class.
        split: ``"train"`` or ``"val"`` (controls augmentation).
    """

    def __init__(self, root: str | Path, split: str = "train") -> None:
        self._base = datasets.ImageFolder(root=str(root))
        self._split = split
        self._img_transform = _aug_transform if split == "train" else _cnn_transform

    # Expose attributes used by the training script
    @property
    def class_to_idx(self) -> dict[str, int]:
        return self._base.class_to_idx

    @property
    def classes(self) -> list[str]:
        return self._base.classes

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return (cnn_image_tensor, handcrafted_feature_tensor, label)."""
        path, label = self._base.samples[idx]
        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil.resize((224, 224), Image.BILINEAR))

        color_feat = extract_hsv_histogram(img_np)
        texture_feat = extract_lbp_texture(img_np)
        handcrafted = np.concatenate([color_feat, texture_feat])

        cnn_tensor = self._img_transform(img_pil)
        feat_tensor = torch.from_numpy(handcrafted)
        return cnn_tensor, feat_tensor, label


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs, feats, labels = zip(*batch)
    return torch.stack(imgs), torch.stack(feats), torch.tensor(labels)


# ---------------------------------------------------------------------------
# Fusion model
# ---------------------------------------------------------------------------


class FusionMoodClassifier(nn.Module):
    """ResNet backbone + handcrafted features → mood class.

    The backbone is frozen; only the fusion MLP is trained.

    Args:
        backbone: Pretrained ResNet model (``resnet50`` or ``resnet18``).
        handcrafted_dim: Dimension of the hand-crafted feature vector.
        num_classes: Number of output mood classes.
        hidden_dim: Hidden size of the fusion MLP.
    """

    def __init__(
        self,
        backbone: nn.Module,
        handcrafted_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()

        # Freeze backbone
        for param in backbone.parameters():
            param.requires_grad = False

        # Strip the classification head → feature extractor
        cnn_out = backbone.fc.in_features if hasattr(backbone.fc, "in_features") else 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.cnn_dim = cnn_out

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + handcrafted_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor, hand_feats: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            images: Batch of normalised image tensors ``(B, 3, 224, 224)``.
            hand_feats: Batch of hand-crafted feature vectors ``(B, D)``.

        Returns:
            Logits ``(B, num_classes)``.
        """
        cnn_feats = self.backbone(images)  # (B, cnn_dim)
        combined = torch.cat([cnn_feats, hand_feats], dim=1)
        return self.fusion(combined)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: FusionMoodClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, feats, labels in tqdm(loader, desc="  train", leave=False):
        imgs = imgs.to(device)
        feats = feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: FusionMoodClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for imgs, feats, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs, feats, labels = imgs.to(device), feats.to(device), labels.to(device)
        outputs = model(imgs, feats)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_labels), np.array(all_preds)


# ---------------------------------------------------------------------------
# Plotting helpers (shared with step 2)
# ---------------------------------------------------------------------------


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss — ResNet + Color/Texture Fusion")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy — ResNet + Color/Texture Fusion")
    ax2.legend()
    ax2.grid(True)
    fig.suptitle("Experiment 2: Color & Texture Feature Fusion", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves → {output_path}")


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title("Confusion Matrix — ResNet + Color/Texture (normalised)", pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix  → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 2: ResNet + color/texture fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=Path("mood_dataset"))
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "resnet18"],
        default="resnet50",
    )
    parser.add_argument(
        "--backbone_ckpt",
        type=Path,
        default=None,
        help="Optional checkpoint from step 2 to initialise backbone weights.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    handcrafted_dim = COLOR_DIM + TEXTURE_DIM  # 96 + 256 = 352

    print(f"Device            : {device}")
    print(f"Backbone          : {args.backbone}")
    print(f"Handcrafted dim   : {handcrafted_dim}  (color={COLOR_DIM}, texture={TEXTURE_DIM})")
    print(f"Epochs            : {args.epochs}\n")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_ds = MoodDatasetWithFeatures(args.data_dir / "train", split="train")
    val_ds = MoodDatasetWithFeatures(args.data_dir / "val", split="val")

    class_names = train_ds.classes
    print(f"Train images : {len(train_ds):,}")
    print(f"Val images   : {len(val_ds):,}")
    print(f"Classes      : {class_names}\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    backbone_fn = getattr(models, args.backbone)
    weights_map = {
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "resnet18": models.ResNet18_Weights.DEFAULT,
    }
    backbone = backbone_fn(weights=weights_map[args.backbone])

    if args.backbone_ckpt is not None and args.backbone_ckpt.exists():
        ckpt = torch.load(args.backbone_ckpt, map_location="cpu")
        # Load backbone weights only (strip the fc layer from step-2 ckpt)
        state = {
            k: v
            for k, v in ckpt["model_state_dict"].items()
            if not k.startswith("fc.")
        }
        backbone.load_state_dict(state, strict=False)
        print(f"Loaded backbone weights from {args.backbone_ckpt}")

    model = FusionMoodClassifier(
        backbone=backbone,
        handcrafted_dim=handcrafted_dim,
        num_classes=NUM_CLASSES,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    writer = SummaryWriter(log_dir=str(args.output_dir / "runs" / "exp2_fusion"))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_acc = 0.0
    ckpt_path = args.checkpoint_dir / "best_model_exp2.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, all_labels, all_preds = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", va_loss, epoch)
        writer.add_scalar("Accuracy/train", tr_acc, epoch)
        writer.add_scalar("Accuracy/val", va_acc, epoch)

        elapsed = time.time() - t0
        marker = " ← best" if va_acc > best_val_acc else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train={tr_loss:.4f}/{tr_acc:.3f}  "
            f"val={va_loss:.4f}/{va_acc:.3f}  "
            f"({elapsed:.1f}s){marker}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "epoch": epoch,
                    "backbone": args.backbone,
                    "class_names": class_names,
                    "handcrafted_dim": handcrafted_dim,
                    "model_state_dict": model.state_dict(),
                    "val_acc": va_acc,
                },
                ckpt_path,
            )

    writer.close()
    print(f"\nBest val accuracy : {best_val_acc:.4f}")
    print(f"Checkpoint saved  → {ckpt_path}\n")

    plot_training_curves(history, args.output_dir / "exp2_training_curves.png")
    plot_confusion_matrix(
        all_labels, all_preds, class_names, args.output_dir / "exp2_confusion_matrix.png"
    )
    print("\nAll done.")


if __name__ == "__main__":
    main()
