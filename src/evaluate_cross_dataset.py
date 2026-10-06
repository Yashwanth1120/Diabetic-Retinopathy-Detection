"""
evaluate_cross_dataset.py
--------------------------
Evaluates an APTOS-trained checkpoint on a DIFFERENT dataset (e.g. IDRiD or
DDR) to test generalization -- i.e. does the model work on data it has never
seen, from a different hospital/population, not just on held-out APTOS images.

This is the experiment that supports a "our model generalizes beyond one
dataset" claim in your paper -- generally a more convincing argument to
reviewers than simply using a newer dataset.

Works with any dataset that has:
    - a folder of images
    - a CSV/table with an image-name column and a DR-grade column
      (0 = No DR ... 4 = Proliferative DR, same convention as APTOS)

IDRiD's "B. Disease Grading" task ships exactly this: an images folder and
a groundtruth CSV with columns "Image name" and "Retinopathy grade".
DDR ships a similar label file. Adjust --image_col / --label_col if a
dataset uses different column names.

Example (IDRiD):
    python src/evaluate_cross_dataset.py \\
        --checkpoint saved_models/best_model.pt \\
        --image_dir idrid/B_Disease_Grading/images \\
        --label_csv idrid/B_Disease_Grading/labels.csv \\
        --image_col "Image name" --label_col "Retinopathy grade" \\
        --image_ext .jpg
"""

import os
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from dataset import get_transforms
from model import build_model


class ExternalDRDataset(Dataset):
    """Generic dataset for an external DR-grading dataset (IDRiD, DDR, etc.)."""

    def __init__(self, image_dir, label_csv, image_col, label_col,
                 image_ext, transform, binary):
        self.image_dir = image_dir
        self.df = pd.read_csv(label_csv)
        self.image_col = image_col
        self.label_col = label_col
        self.image_ext = image_ext
        self.transform = transform
        self.binary = binary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = str(row[self.image_col]).strip()
        if not name.lower().endswith(self.image_ext.lower()):
            name = name + self.image_ext

        img_path = os.path.join(self.image_dir, name)
        image = Image.open(img_path).convert("RGB")

        label = int(row[self.label_col])
        if self.binary:
            label = 0 if label == 0 else 1

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="saved_models/best_model.pt",
                         help="Checkpoint trained on APTOS (or any source dataset)")
    parser.add_argument("--image_dir", type=str, required=True,
                         help="Folder containing the external dataset's images")
    parser.add_argument("--label_csv", type=str, required=True,
                         help="CSV with image names + DR grade labels")
    parser.add_argument("--image_col", type=str, default="Image name")
    parser.add_argument("--label_col", type=str, default="Retinopathy grade")
    parser.add_argument("--image_ext", type=str, default=".jpg",
                         help="Image file extension if not already in the CSV names")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="saved_models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    binary = ckpt["binary"]
    backbone = ckpt["backbone"]
    num_classes = ckpt["num_classes"]
    img_size = ckpt["img_size"]

    dataset = ExternalDRDataset(
        image_dir=args.image_dir,
        label_csv=args.label_csv,
        image_col=args.image_col,
        label_col=args.label_col,
        image_ext=args.image_ext,
        transform=get_transforms(train=False, img_size=img_size),
        binary=binary,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Loaded {len(dataset)} external images for cross-dataset evaluation.")

    model = build_model(backbone, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    target_names = ["No DR", "DR"] if binary else \
        ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print("\n=== Cross-Dataset Generalization Report ===")
    print(f"Trained on: APTOS (source) | Evaluated on: external dataset at {args.image_dir}")
    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Cross-Dataset Confusion Matrix (External Test Set)")
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, "confusion_matrix_cross_dataset.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nConfusion matrix saved to: {out_path}")

    print(
        "\nTip for your paper: report this accuracy/F1 side-by-side with your "
        "in-dataset (APTOS validation) numbers from evaluate.py. A smaller gap "
        "between the two is evidence the model generalizes rather than overfits."
    )


if __name__ == "__main__":
    main()
