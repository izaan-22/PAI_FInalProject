"""Compose image features and produce objects in bounded batches."""
from pathlib import Path
import hashlib
import pandas as pd
import numpy as np
from .models import Fruit, Vegetable
from ..abdul_haseeb.image_loader import CorruptedImageError, load_image
from .vector_engine import preprocess, features


class BatchProcessor:
    def __init__(self, batch_id, size, gain):
        self.batch_id, self.size, self.gain = batch_id, size, gain
        self.items, self.errors = [], []
        self.image_matrix = None

    def load_images_from_directory(self, rows, directory=None):
        images, indices = [], []
        for i, row in rows.iterrows():
            path = Path(directory or ".") / row.image
            try:
                images.append(load_image(path, self.size))
                indices.append(i)
            except CorruptedImageError as exc:
                self.errors.append({"item_id": row.item_id, "error": str(exc)})
        self.image_matrix = np.stack(images) if images else None
        return rows.loc[indices].reset_index(drop=True)

    def preprocess_batch(self):
        return preprocess(self.image_matrix, self.gain)

    def compute_vectorized_stats(self):
        return features(*self.preprocess_batch())

    def process(self, rows):
        pieces = []
        for start in range(0, len(rows), 32):
            metadata = self.load_images_from_directory(rows.iloc[start:start+32])
            if metadata.empty:
                continue
            values, ranges = self.compute_vectorized_stats()
            details = []
            for row, interval, img in zip(metadata.itertuples(), ranges, self.image_matrix):
                def measured(key):
                    value = getattr(row, key)
                    return None if pd.isna(value) else value
                base = (row.item_id, row.name, row.batch_code, measured("weight_gram"))
                if row.category == "fruit":
                    item = Fruit(*base, measured("brix_sweetness"), tuple(interval), measured("acidity_percent"))
                    extra = {"sugar_acid_ratio": item.get_sugar_acid_ratio()}
                else:
                    item = Vegetable(*base, measured("firmness_index"), measured("foliage_health_score"))
                    extra = {"crispness": item.check_crispness_level()}
                self.items.append(item)
                details.append(extra | {"rule_grade": item.calculate_grade(), "quality_score": item.quality_score,
                                        "image_hash": hashlib.sha256(img.tobytes()).hexdigest()})
            pieces.append(pd.concat([metadata, values, pd.DataFrame(details)], axis=1))
        if not pieces:
            raise ValueError("No readable images. Check the paths in the manifest.")
        return pd.concat(pieces, ignore_index=True), list(values.columns)
