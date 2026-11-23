"""
Train a research-only anatomical-plane classifier from a CSV manifest.

Before training:
1. Extract FETAL_PLANES_DB under data/raw/.
2. Run: python -m src.models.dataset_inspect
3. Set CSV_PATH, IMAGE_COLUMN, LABEL_COLUMN and optional PATIENT_COLUMN below
   to match the real metadata columns shown by the inspection script.
4. Confirm the dataset's official patient-level split column if available.

This script deliberately refuses to silently create image-level random splits when
a patient ID column is available. It is NOT a Spina Bifida detector.
"""
from pathlib import Path
import json
import argparse
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

class UltrasoundDataset(Dataset):
    def __init__(self, frame, image_root, label_to_idx, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        path = self.image_root / str(row["_image"])
        image = Image.open(path).convert("RGB")
        return self.transform(image), self.label_to_idx[str(row["_label"])]

def resolve_column(df, requested, candidates, kind):
    if requested and requested in df.columns:
        return requested
    lower = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(
        f"Could not identify {kind} column. Available columns: {list(df.columns)}. "
        f"Pass --{kind}-column with the exact column name."
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--image-root", required=True, help="Directory containing images")
    parser.add_argument("--image-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--patient-column", default="Patient_num")
    parser.add_argument("--split-column", default="Train")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="models/anatomical_plane_mobilenetv2.pth")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    image_root = Path(args.image_root)
    if not image_root.is_absolute():
        image_root = ROOT / image_root

    # FETAL_PLANES_DB is semicolon-delimited. Strip header whitespace (e.g. "Train ").
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [str(c).strip() for c in df.columns]
    image_col = resolve_column(df, args.image_column,
        ["Image_name", "image_name", "filename", "image", "file_name"], "image")
    label_col = resolve_column(df, args.label_column,
        ["Plane", "plane", "label", "class", "Brain_plane"], "label")

    patient_col = args.patient_column.strip() if args.patient_column else None
    if not patient_col or patient_col not in df.columns:
        raise ValueError(f"Required patient column '{patient_col}' not found. Available: {list(df.columns)}")
    split_col = args.split_column.strip()
    if split_col not in df.columns:
        raise ValueError(f"Required official split column '{split_col}' not found. Available: {list(df.columns)}")

    df = df.dropna(subset=[image_col, label_col]).copy()
    def image_filename(value):
        name = Path(str(value).strip()).name
        return name if Path(name).suffix else name + ".png"
    df["_image"] = df[image_col].astype(str).map(image_filename)
    df["_label"] = df[label_col].astype(str).str.strip()
    existing = {p.name for p in image_root.rglob("*") if p.is_file()}
    df = df[df["_image"].isin(existing)].copy()
    if df.empty:
        raise ValueError(
            "No metadata rows matched image files. Check --image-root and metadata filename column. "
            "Run dataset_inspect.py and inspect the archive layout."
        )

    labels = sorted(df["_label"].unique())
    if len(labels) < 2:
        raise ValueError(f"Need at least two classes; found {labels}")
    label_to_idx = {label: i for i, label in enumerate(labels)}

    # Use the dataset's published split exactly: Train=1 is train, Train=0 is test.
    split_values = df[split_col].astype(str).str.strip()
    if not set(split_values.unique()).issubset({"0", "1"}):
        raise ValueError(f"Expected only 0/1 in official split column; found {sorted(split_values.unique())}")
    train_df = df[split_values == "1"].copy()
    test_df = df[split_values == "0"].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Official split produced an empty train or test partition.")
    train_patients = set(train_df[patient_col].astype(str))
    test_patients = set(test_df[patient_col].astype(str))
    overlap = train_patients & test_patients
    if overlap:
        raise RuntimeError(f"Patient leakage in official split: {len(overlap)} overlapping patients. Stopping.")
    split_method = f"official_{split_col}_1_train_0_test_patient_disjoint"
    print(f"Official split: train={len(train_df)} rows / {len(train_patients)} patients; "
          f"test={len(test_df)} rows / {len(test_patients)} patients")
    print("Training class counts:\n", train_df["_label"].value_counts().to_string())
    print("Test class counts:\n", test_df["_label"].value_counts().to_string())

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = UltrasoundDataset(train_df, image_root, label_to_idx, train_tf)
    test_ds = UltrasoundDataset(test_df, image_root, label_to_idx, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, len(labels))
    model.to(device)

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)
        print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/max(total,1):.4f} accuracy={correct/max(total,1):.4f}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(targets.numpy().tolist())

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Balanced accuracy:", balanced_accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "classes": labels,
        "architecture": "mobilenet_v2",
        "input_size": 224,
        "split_method": split_method,
        "patient_column": patient_col,
        "train_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "warning": "Research-only anatomical-plane classifier; not a Spina Bifida detector.",
    }, output)
    print(f"Saved model bundle: {output}")

if __name__ == "__main__":
    main()
