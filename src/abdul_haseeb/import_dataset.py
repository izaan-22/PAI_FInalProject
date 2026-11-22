"""Build a reproducible manifest from an extracted Kaggle image tree."""
import argparse
import csv
import os
from pathlib import Path
import random
import re
from tempfile import NamedTemporaryFile, TemporaryDirectory
from collections import defaultdict
from PIL import Image, ImageOps
from .data_handler import CROPS, IMAGE_SUFFIXES

DATASET = "raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables"
MANIFEST_COLUMNS = ["item_id", "name", "category", "batch_code", "image", "grade",
                    "original_path", "source_id"]

def identify(path):
    parts = [re.sub("[^a-z]", "", p.lower()) for p in path.parts[:-1]]
    parts = [p.replace("bittergroud", "bittergourd").replace("tamto", "tomato")
             .replace("bellpepper", "capsicum") for p in parts]
    for i in range(len(parts)-1, -1, -1):
        for prefix, grade in [("fresh","A"),("rotten","C"),("stale","C"),("choice","B")]:
            part = parts[i]
            if not part.startswith(prefix):
                continue
            nearby = [part[len(prefix):], *parts[max(0,i-1):i], *parts[i+1:i+2]]
            for crop in CROPS:
                if any(word.startswith(crop) or (crop=="tomato" and word=="tamato") for word in nearby):
                    return crop, grade
    return None


def image_pools(source):
    if not source.is_dir():
        raise ValueError(f"Extract the image folders first: {source}")
    pools = defaultdict(list)
    for p in sorted(source.rglob("*")):
        relative = p.relative_to(source)
        if (p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
                and not any(part.startswith(".") or part == "__MACOSX" for part in relative.parts)):
            key = identify(relative)
            if key:
                pools[key].append(p)
    if not pools:
        raise ValueError("No recognized crop folders, e.g. freshapples or rotten/tomato.")
    rng = random.Random(59)
    for paths in pools.values():
        rng.shuffle(paths)
    return pools


def balanced_images(pools):
    while pools:
        for key in list(pools):
            if pools[key]:
                yield key, pools[key].pop()
            else:
                del pools[key]


def import_manifest(source, out, limit=None):
    """Reference the original photos without copying or resizing them."""
    if limit is not None and limit < 1:
        raise ValueError("Supply a positive limit.")
    source = source.resolve()
    out = out.resolve()
    pools = image_pools(source)
    out.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile("w", dir=out, newline="", encoding="utf8", delete=False)
    temp = Path(stream.name)
    try:
        with stream:
            writer = csv.writer(stream)
            writer.writerow(MANIFEST_COLUMNS)
            for count, ((crop, grade), path) in enumerate(balanced_images(pools), start=1):
                relative = path.relative_to(source)
                # Keep alternate file formats with the same folder and stem in one evaluation group.
                group = relative.with_suffix("").as_posix().casefold()
                writer.writerow([f"kg-{count:05d}", crop, CROPS[crop], "kaggle",
                                 os.path.relpath(path, out), grade, relative.as_posix(), group])
                if count == limit:
                    break
        os.replace(temp, out / "metadata.csv")
    finally:
        temp.unlink(missing_ok=True)
    print(f"Indexed {count:,} photos in {out / 'metadata.csv'}.")


def import_images(source, out, limit):
    if limit < 1:
        raise ValueError("Supply a positive limit.")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Use an empty destination: {out}")
    pools = image_pools(source)
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as temp:
        dest = Path(temp)
        (dest/"images").mkdir()
        with (dest/"metadata.csv").open("w", newline="", encoding="utf8") as stream:
            writer = csv.writer(stream)
            writer.writerow(MANIFEST_COLUMNS)
            count = 0
            for (crop, grade), path in balanced_images(pools):
                try:
                    with Image.open(path) as im:
                        thumb = ImageOps.exif_transpose(im).convert("RGB")
                        thumb.thumbnail((128,128),Image.Resampling.LANCZOS)
                except (OSError,ValueError):
                    continue
                tag = f"kg-{count+1:05d}"
                rel = f"images/{tag}.png"
                thumb.save(dest/rel)
                writer.writerow([tag,crop,CROPS[crop],"kaggle",rel,grade,path.relative_to(source), ""])
                count += 1
                if count == limit:
                    break
        if not count:
            raise ValueError("All matched images were unreadable.")
        for p in dest.iterdir():
            os.replace(p,out/p.name)
    print(f"Imported {count} images into {out}. Review labels and source groups before evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Extracted Kaggle directory. Omit to download with kagglehub.")
    parser.add_argument("--out",type=Path,default=Path("data/raw"))
    parser.add_argument("--manifest-only", action="store_true", help="Index original files in place.")
    parser.add_argument("--limit", type=int, help="Maximum photos (default: all in place, 1500 for copies).")
    args = parser.parse_args()
    if args.source is None:
        try:
            import kagglehub
        except ImportError:
            parser.exit(2,"Install optional downloader with: python -m pip install kagglehub\n")
        args.source = Path(kagglehub.dataset_download(DATASET))
    if args.manifest_only:
        import_manifest(args.source, args.out, args.limit)
    else:
        import_images(args.source, args.out, 1500 if args.limit is None else args.limit)
