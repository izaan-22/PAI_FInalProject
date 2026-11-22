"""Validate manifests and impute weights without inventing inspection labels."""
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd

CROPS = {"apple": "fruit", "banana": "fruit", "orange": "fruit", "bittergourd": "vegetable",
         "capsicum": "vegetable", "tomato": "vegetable", "potato": "vegetable", "okra": "vegetable",
         "cucumber": "vegetable", "carrot": "vegetable", "mango": "fruit", "strawberry": "fruit"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MEASUREMENTS = {"weight_gram": (0, None), "brix_sweetness": (0, 100),
                "acidity_percent": (0, 100), "firmness_index": (0, 1),
                "foliage_health_score": (0, 1)}


def inspection_manifest(source, crop, batch, output):
    """Create an unlabeled manifest referencing one photo or a folder of photos."""
    crop = "".join(crop.casefold().replace("-", " ").split())
    if crop == "bellpepper":
        crop = "capsicum"
    if crop not in CROPS:
        raise ValueError(f"Choose a supported crop: {', '.join(sorted(CROPS))}.")
    source = source.expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Photo or folder not found: {source}")
    candidates = sorted(source.iterdir()) if source.is_dir() else [source]
    photos = [path for path in candidates if not path.name.startswith(".")
              and path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()]
    if not photos:
        raise ValueError(f"No JPG, JPEG, PNG, BMP or WebP photos found in {source}.")
    default_batch = source.name if source.is_dir() else source.stem
    batch = (batch or "").strip() or default_batch.strip() or "inspection"
    output.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=output, prefix="inspection-", suffix=".csv",
                            newline="", encoding="utf8", delete=False) as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "name", "category", "batch_code", "image"])
        for index, photo in enumerate(photos, start=1):
            writer.writerow([f"photo-{index:04d}", crop, CROPS[crop], batch, photo])
    return Path(stream.name)


def read_metadata(path):
    path = Path(path).resolve()
    df = pd.read_csv(path, dtype=str).apply(lambda s: s.str.strip()).replace("", np.nan)
    required = {"item_id", "name", "category", "batch_code", "image"}
    missing = required - set(df)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    if df.empty or df[list(required)].isna().any().any() or df.item_id.duplicated().any():
        raise ValueError("Each row needs a unique item_id and complete identifying fields.")
    df["category"] = df.category.str.lower()
    if not df.category.isin(["fruit", "vegetable"]).all():
        raise ValueError("Category must be fruit or vegetable.")
    df["image"] = df.image.map(lambda p: str((path.parent / p).resolve()))
    for col, (low, high) in MEASUREMENTS.items():
        if col not in df:
            df[col] = np.nan
        values = pd.to_numeric(df[col], errors="coerce")
        valid = np.isfinite(values) & (values >= low)
        if col == "weight_gram":
            valid &= values > 0
        if high is not None:
            valid &= values <= high
        if (df[col].notna() & ~valid).any():
            raise ValueError(f"Invalid {col} value.")
        df[col] = values
    df["weight_imputed"] = df.weight_gram.isna()
    df["weight_gram"] = df.weight_gram.fillna(df.groupby(["category","name"]).weight_gram.transform("median"))
    df["weight_imputed"] &= df.weight_gram.notna()
    return df
