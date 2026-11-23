# FETAL_PLANES_DB anatomical-plane model training

## Verified dataset mapping

The inspected archive contains 12,400 PNG images and 12,400 metadata rows. The CSV columns are `Image_name`, `Patient_num`, `Plane`, `Brain_plane`, `Operator`, `US_Machine`, and `Train` (the original header may contain trailing whitespace, which the training script strips).

For the first model, predict `Plane` (six classes). `Brain_plane` is a separate five-label field and can be used for a later brain-plane-only experiment. This dataset does not provide Spina Bifida disease labels.

The official split is `Train=1` for training (7,129 rows) and `Train=0` for testing (5,271 rows). The dataset inspection confirmed 1,792 patients and no patient IDs crossing these two splits. The updated training script uses this official split and stops if it detects patient overlap; it does not create a random split.

## Run from the project root (Windows PowerShell)

```powershell
python -m src.models.train_classifier --csv "data/raw/FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv" --image-root "data/raw/FETAL_PLANES_ZENODO/Images" --image-column "Image_name" --label-column "Plane" --patient-column "Patient_num" --split-column "Train" --epochs 8 --batch-size 32
```

The script reports class distributions, per-class precision/recall/F1, confusion matrix, accuracy, balanced accuracy, and macro F1. It saves `models/anatomical_plane_mobilenetv2.pth`.

## Compute recommendation

Use Google Colab with a GPU for training rather than the Windows laptop's integrated Intel graphics. The script may run on CPU, but training will be much slower. Keep the original official split intact.

## Scope and safety

This is a research-only anatomical-plane classifier. It is not a Spina Bifida detector, diagnostic system, or medical device. Do not interpret its predictions as clinical findings.
