# Automated Diabetic Retinopathy Detection Using Deep Learning

A CNN-based system (with transfer learning) that classifies retinal fundus
images by diabetic retinopathy (DR) severity, with Grad-CAM explainability
and a Streamlit demo app.

## 1. Project Structure

```
dr_project/
├── README.md
├── requirements.txt
├── download_data.py           # downloads APTOS via kagglehub into data/
├── quick_test.py               # sanity-checks the data pipeline before training
├── data/                     # put the APTOS dataset here (not included)
│   ├── train_images/
│   └── train.csv
├── saved_models/             # trained model checkpoints land here
├── src/
│   ├── dataset.py            # dataset loading + preprocessing + class balancing
│   ├── model.py               # model definition (transfer learning backbone)
│   ├── train.py                # training loop
│   ├── evaluate.py            # metrics: accuracy, F1, confusion matrix
│   ├── evaluate_cross_dataset.py  # test generalization on IDRiD/DDR
│   └── gradcam.py             # Grad-CAM implementation
└── app/
    └── app.py                 # Streamlit demo: upload image -> prediction + heatmap
```

## 2. Get the dataset

1. Create a free Kaggle account if you don't have one.
2. Go to the **APTOS 2019 Blindness Detection** competition page and download
   `train.csv` and `train_images.zip`.
3. Unzip so you end up with:
   ```
   data/train.csv
   data/train_images/<id_code>.png
   ```

   Or use the included helper script, which does the download and copy for
   you via `kagglehub` (requires `kaggle.json` credentials set up first):
   ```bash
   pip install kagglehub
   python download_data.py
   ```

`train.csv` has two columns: `id_code` (image filename without extension) and
`diagnosis` (0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR).

**Before training, run the sanity check:**
```bash
python quick_test.py
```
This confirms the data pipeline loads correctly on a small batch before you
commit to a full (multi-hour) training run.

> **If you don't have a GPU on your laptop**: use Google Colab (free GPU) or
> Kaggle Notebooks (free GPU, and the dataset is already attachable directly
> from Kaggle without downloading). This is the easiest path for most
> students — don't fight with local CUDA setup if you don't need to.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Train the model

```bash
python src/train.py --data_dir data --epochs 15 --backbone efficientnet_b0 --binary false
```

Key flags:
- `--backbone`: `efficientnet_b0` or `resnet50` (try both — this comparison
  is useful material for your paper later)
- `--binary`: set to `true` to collapse to a 2-class problem (No DR vs DR) if
  5-class training is unstable or accuracy is poor — this is a legitimate,
  commonly-used simplification (see README note at the bottom)
- `--epochs`: start with 10-15; increase if validation accuracy is still
  climbing at the end

This saves the best model (by validation F1) to `saved_models/best_model.pt`.

## 5. Evaluate

```bash
python src/evaluate.py --data_dir data --checkpoint saved_models/best_model.pt
```

Prints accuracy, per-class precision/recall/F1, and a confusion matrix image
saved to `saved_models/confusion_matrix.png`.

## 6. Test generalization on a second dataset (for the paper)

Training and testing only on APTOS shows your model works on APTOS. For a
stronger, more "real-world accurate" claim in your paper, evaluate the same
APTOS-trained model on a second, independent dataset it has never seen —
e.g. **IDRiD** (Indian Diabetic Retinopathy Image Dataset) or **DDR**. This
shows the model generalizes across hospitals/populations rather than just
overfitting to one data source.

1. Download IDRiD's "B. Disease Grading" task (freely available) — it ships
   an images folder and a groundtruth CSV with columns `Image name` and
   `Retinopathy grade`, the same 0-4 grading convention as APTOS.
2. Run:
   ```bash
   python src/evaluate_cross_dataset.py \
       --checkpoint saved_models/best_model.pt \
       --image_dir idrid/B_Disease_Grading/images \
       --label_csv idrid/B_Disease_Grading/labels.csv \
       --image_col "Image name" --label_col "Retinopathy grade" \
       --image_ext .jpg
   ```
3. Compare this accuracy/F1 against your in-dataset (APTOS validation)
   numbers from `evaluate.py`. Report both side by side in your paper — a
   smaller gap between the two is evidence of genuine generalization, which
   is a stronger and more defensible claim than simply using a newer dataset.

This script works with DDR or any similarly-structured dataset too — just
point `--image_col`/`--label_col` at whatever column names that dataset's
label file actually uses.

## 7. Run the demo app

```bash
streamlit run app/app.py
```

Upload a retinal image, get a predicted severity class and a Grad-CAM heatmap
overlay showing which regions influenced the prediction.

## 7. Suggested week-by-week use of this code

- **Week 3**: get `train.py` running end-to-end on a small subset (e.g. 500
  images) just to confirm the pipeline works, before training on the full
  dataset
- **Week 4**: full training run, try both backbones, record metrics for
  comparison
- **Week 5**: `gradcam.py` is already wired in — no extra work needed beyond
  generating and reviewing sample heatmaps
- **Week 6**: `app/app.py` is your demo — customize the UI text/title as you
  like

## Note on binary vs. 5-class framing

If 5-class severity classification proves hard to get good accuracy on
(common — adjacent classes like Mild/Moderate are visually similar and the
dataset is imbalanced), switching to binary (No DR vs. DR) is a legitimate
fallback used in published work, not a "cheat." You can mention this as a
deliberate design decision in your report, not a failure.
