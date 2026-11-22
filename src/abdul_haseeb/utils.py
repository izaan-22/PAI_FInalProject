"""Small file and configuration helpers."""
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf8") as f:
        temp = Path(f.name)
        try:
            json.dump(value, f, indent=2, allow_nan=False)
        except (TypeError, ValueError):
            f.close()
            temp.unlink()
            raise
    os.replace(temp, path)


def read_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf8"))
    if cfg["image_size"] not in (64, 128):
        raise ValueError("image_size must be 64 or 128.")
    gain = cfg["channel_gain"]
    if len(gain) != 3 or not all(0 < x <= 3 for x in gain):
        raise ValueError("Provide three channel gains above 0 and at most 3.")
    for key, limit in [("latitude",90), ("longitude",180)]:
        if not math.isfinite(cfg[key]) or abs(cfg[key]) > limit:
            raise ValueError(f"Invalid {key}.")
    for key in ["market", "commodity", "price_type"]:
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"Specify {key}.")
    for key in ["training_csv", "output_dir"]:
        cfg[key] = path.parent / cfg[key]
    if not isinstance(cfg.get("exclude_conflicting_labels", False), bool):
        raise ValueError("exclude_conflicting_labels must be true or false.")
    inspection = cfg.get("inspection_csv")
    cfg["inspection_csv"] = path.parent / inspection if inspection else None
    cfg["cache"] = path.parent / "data/cache/market_weather_cache.json"
    return cfg
