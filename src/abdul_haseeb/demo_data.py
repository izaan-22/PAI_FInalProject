"""Create seeded, explicitly synthetic fixtures for a network-free lab demonstration."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]/"data/demo"


def generate():
    rng = np.random.default_rng(59)
    y,x = np.mgrid[:64,:64]
    (ROOT/"images").mkdir(parents=True,exist_ok=True)
    for split, repeats in [("train",24),("inspect",4)]:
        with (ROOT/f"{split}.csv").open("w",newline="") as f:
            w = csv.writer(f)
            w.writerow("item_id name category batch_code image grade source_id weight_gram brix_sweetness acidity_percent firmness_index foliage_health_score".split())
            for n in range(repeats):
                for crop, cat, color in [("apple","fruit",[200,60,40]),("capsicum","vegetable",[65,168,75])]:
                    for j,grade in enumerate("ABC"):
                        tag = f"{split}-{crop}-{n:02}-{j}"
                        cx,cy = rng.integers(28,36,2)
                        rx,ry = rng.integers(18,27,2)
                        dist = ((x-cx)/rx)**2+((y-cy)/ry)**2
                        mask = dist<1
                        pixels = np.full((64,64,3), rng.uniform(211,247))
                        tone = np.array(color)+rng.normal(0,9,3)
                        pixels[mask] = tone * (1-.18*dist[mask,None])
                        for _ in range([1,5,14][j]):
                            a,b = rng.normal([cx,cy],[9,9])
                            radius = rng.uniform(*[(1,3),(3,6),(5,9)][j])
                            pixels[mask & ((x-a)**2+(y-b)**2<radius**2)] = rng.uniform([47,29,14],[95,59,32])
                        pixels += rng.normal(0,3,pixels.shape)
                        path = f"images/{tag}.png"
                        Image.fromarray(pixels.clip(0,255).astype("uint8")).save(ROOT/path)
                        w.writerow([tag,crop,cat,f"{split}-{n%3+1}",path,grade if split=="train" else "",
                                    tag,round(rng.uniform(100,240),1) if n%7 else "",
                                    [14,10,7][j] if cat=="fruit" else "",.6 if cat=="fruit" else "",
                                    [1,.7,.4][j] if cat=="vegetable" else "",[.95,.7,.4][j] if cat=="vegetable" else ""])


if __name__ == "__main__":
    generate()
