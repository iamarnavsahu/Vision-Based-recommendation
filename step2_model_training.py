"""Step 2: Model Training — Experiment 1 (ResNet Baseline)
----------------------------------------------------------
Trains a ResNet-50 (or ResNet-18) mood classifier on the Places365-derived
mood dataset produced by step1_dataset_preparation.py.

Experiments
-----------
    Exp 1 (default): ResNet-50 pretrained on ImageNet, fine-tuned end-to-end.
    Exp 3 (--backbone resnet18): Same setup with the lighter ResNet-18.

Usage
-----
    # Experiment 1: ResNet-50 baseline
    python step2_model_training.py

    # Experiment 3: smaller backbone
    python step2_model_training.py --backbone resnet18

    # Custom settings
    python step2_model_training.py \\
        --data_dir mood_dataset \\
        --backbone resnet50 \\
        --epochs 30 \\
        --lr 1e-3 \\
        --batch_size 64 \\
        --output_dir results \\
        --checkpoint_dir checkpoints

Outputs
-------
    checkpoints/best_model.pth          — best validation accuracy weights
    results/exp1_training_curves.png    — loss & accuracy over epochs
    results/exp1_confusion_matrix.png   — per-class confusion matrix
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
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

# ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------


def get_transforms(split: str) -> transforms.Compose:
    """Return the image transforms for training or validation.

    Args:
        split: ``"train"`` or ``"val"``.

    Returns:
        A composed torchvision transform pipeline.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a ResNet classifier with a custom final layer.

    Args:
        backbone: ``"resnet50"`` or ``"resnet18"``.
        num_classes: Number of output mood classes.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        A PyTorch ``nn.Module`` ready for training.

    Raises:
        ValueError: If an unsupported backbone name is provided.
    """
    weights_map = {
        "resnet50": models.ResNet50_Weights.DEFAULT if pretrained else None,
        "resnet18": models.ResNet18_Weights.DEFAULT if pretrained else None,
    }
    if backbone not in weights_map:
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose resnet50 or resnet18.")

    model_fn = getattr(models, backbone)
    model = model_fn(weights=weights_map[backbone])

    # Replace the classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch.

    Args:
        model: The classifier.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Gradient-descent optimizer.
        device: Torch device.

    Returns:
        Tuple of (average loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate the model on a validation set.

    Args:
        model: The classifier.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        Tuple of (average loss, accuracy, all_labels, all_preds).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for images, labels in tqdm(loader, desc="  val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_labels),
        np.array(all_preds),
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: Path,
    backbone: str,
) -> None:
    """Save a two-panel loss / accuracy plot.

    Args:
        history: Dict with keys ``train_loss``, ``val_loss``,
                 ``train_acc``, ``val_acc``.
        output_path: Where to save the PNG.
        backbone: Backbone name (used in the plot title).
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"{backbone} — Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{backbone} — Accuracy")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(f"Mood Classifier Training — {backbone}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves → {output_path}")


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    output_path: Path,
    backbone: str,
) -> None:
    """Save a normalised confusion matrix heatmap.

    Args:
        labels: Ground-truth integer class indices.
        preds: Predicted integer class indices.
        class_names: Human-readable class names (in label index order).
        output_path: Where to save the PNG.
        backbone: Backbone name (used in the plot title).
    """
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title(f"Confusion Matrix — {backbone} (normalised)", pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix  → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ResNet mood classifier (Experiments 1 & 3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("mood_dataset"),
        help="Directory with train/ and val/ splits from step 1.",
    )
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "resnet18"],
        default="resnet50",
        help="CNN backbone: resnet50 (Exp 1) or resnet18 (Exp 3).",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    # Determine experiment tag for file names
    exp_tag = "exp1" if args.backbone == "resnet50" else "exp3"

    # Dirs
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")
    print(f"Backbone : {args.backbone}")
    print(f"Epochs   : {args.epochs}")
    print(f"LR       : {args.lr}")
    print(f"Batch    : {args.batch_size}\n")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_dataset = datasets.ImageFolder(
        root=str(args.data_dir / "train"),
        transform=get_transforms("train"),
    )
    val_dataset = datasets.ImageFolder(
        root=str(args.data_dir / "val"),
        transform=get_transforms("val"),
    )

    # ImageFolder sorts class names alphabetically; store the mapping
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print(f"Train images : {len(train_dataset):,}")
    print(f"Val images   : {len(val_dataset):,}")
    print(f"Classes      : {class_names}\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model, loss, optimiser, scheduler
    # ------------------------------------------------------------------
    model = build_model(
        backbone=args.backbone,
        num_classes=NUM_CLASSES,
        pretrained=not args.no_pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter(log_dir=str(args.output_dir / "runs" / f"{exp_tag}_{args.backbone}"))

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
    checkpoint_path = args.checkpoint_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, all_labels, all_preds = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        elapsed = time.time() - t0
        marker = " ← best" if val_acc > best_val_acc else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"({elapsed:.1f}s){marker}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "backbone": args.backbone,
                    "class_names": class_names,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )

    writer.close()

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved → {checkpoint_path}\n")

    # ------------------------------------------------------------------
    # Plots (use labels/preds from the final epoch for the conf. matrix)
    # ------------------------------------------------------------------
    curves_path = args.output_dir / f"{exp_tag}_training_curves.png"
    plot_training_curves(history, curves_path, args.backbone)

    cm_path = args.output_dir / f"{exp_tag}_confusion_matrix.png"
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_path, args.backbone)

    print("\nAll done.")


if __name__ == "__main__":
    main()
