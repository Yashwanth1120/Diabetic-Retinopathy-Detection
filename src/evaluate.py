"""
evaluate.py
-----------
Loads a trained checkpoint and reports accuracy, per-class precision/recall/F1,
and a confusion matrix (saved as an image) on the validation split.

Example:
    python src/evaluate.py --data_dir data --checkpoint saved_models/best_model.pt
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_dataloaders
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="saved_models/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    binary = ckpt["binary"]
    backbone = ckpt["backbone"]
    num_classes = ckpt["num_classes"]
    img_size = ckpt["img_size"]

    _, val_loader, _ = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, img_size=img_size, binary=binary,
    )

    model = build_model(backbone, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    if binary:
        target_names = ["No DR", "DR"]
    else:
        target_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Diabetic Retinopathy Classification")
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(args.checkpoint), "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nConfusion matrix saved to: {out_path}")


if __name__ == "__main__":
    main()
