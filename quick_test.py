"""
quick_test.py
-------------
Sanity check: confirms the dataset loads correctly end-to-end before you
commit to a full training run. Run this first after downloading the data.

Run from the project root:
    python quick_test.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from dataset import get_dataloaders  # noqa: E402


def main():
    print("Loading a small batch from the dataset...")
    train_loader, val_loader, num_classes = get_dataloaders("data", batch_size=8)

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Num classes: {num_classes}")
    print(f"\nTrain batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print("\nIf you see a batch shape like torch.Size([8, 3, 224, 224]) above, "
          "the data pipeline works. You're ready to run src/train.py.")


if __name__ == "__main__":
    main()
