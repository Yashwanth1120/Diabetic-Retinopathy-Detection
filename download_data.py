"""
download_data.py
-----------------
Downloads the APTOS 2019 Blindness Detection dataset via kagglehub and
copies it into the local data/ folder with the structure train.py expects.

Prerequisites:
    1. A Kaggle account, with the competition rules accepted on the
       APTOS 2019 Blindness Detection competition page.
    2. kaggle.json API credentials set up (see README.md section 2).
    3. pip install kagglehub

Run:
    python download_data.py
"""

import os
import shutil
import kagglehub


def main():
    print("Downloading APTOS 2019 Blindness Detection dataset...")
    cache_path = kagglehub.competition_download("aptos2019-blindness-detection")

    print(f"Downloaded to cache: {cache_path}")
    print("Copying into local data/ folder...")
    shutil.copytree(cache_path, "data", dirs_exist_ok=True)

    data_path = os.path.abspath("data")
    print(f"\nDataset ready at: {data_path}")

    # Quick structure check
    expected = ["train.csv", "train_images"]
    found = os.listdir("data")
    missing = [e for e in expected if e not in found]
    if missing:
        print(f"\nWARNING: expected items not found at top level: {missing}")
        print(f"Actual contents of data/: {found}")
        print("You may need to adjust paths in src/dataset.py if the "
              "download has an extra nested folder.")
    else:
        print("Structure looks correct: train.csv and train_images/ both found.")


if __name__ == "__main__":
    main()
