"""
train.py
--------
Trains the DR classifier with transfer learning.

Example:
    python src/train.py --data_dir data --epochs 15 --backbone efficientnet_b0
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model


def str2bool(v):
    return str(v).lower() in ("true", "1", "yes")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="val", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return val_loss, val_acc, val_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                         choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--binary", type=str2bool, default=False)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--freeze_backbone", type=str2bool, default=False)
    parser.add_argument("--output_dir", type=str, default="saved_models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, num_classes = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, img_size=args.img_size,
        binary=args.binary,
    )
    print(f"Classes: {num_classes} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(args.backbone, num_classes=num_classes,
                         freeze_backbone=args.freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "num_classes": num_classes,
                "binary": args.binary,
                "img_size": args.img_size,
                "val_f1": val_f1,
            }, checkpoint_path)
            print(f"  -> New best model saved (val_f1={val_f1:.4f})")

    print(f"\nTraining complete. Best val_f1: {best_f1:.4f}")
    print(f"Checkpoint saved at: {os.path.join(args.output_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
