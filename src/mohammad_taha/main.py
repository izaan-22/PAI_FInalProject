"""Interactive CLI and reproducible command-line entry point."""
import argparse
import json
from pathlib import Path
from ..abdul_haseeb.api import MarketDataFetcher
from ..abdul_haseeb.data_handler import CROPS, inspection_manifest
from ..abdul_haseeb.utils import read_config
from ..izaan_aamir.integration import Project
from .terminal import context_report, heading, inspection_report

ROOT = Path(__file__).resolve().parents[2]
ERRORS = (OSError, ValueError, KeyError, TypeError)


def choose_path(label, default=None):
    if default is None:
        value = input(f"  {label}> ").strip()
        if not value:
            raise ValueError(f"{label} is required.")
        return Path(value).expanduser().resolve()
    print(f"  Default  {default}")
    val = input(f"  {label} (Enter for default)> ").strip()
    return Path(val).expanduser().resolve() if val else default


def inspect_images(project, source, crop, batch, offline):
    manifest = inspection_manifest(source, crop, batch, project.root / "inputs")
    print(f"  Inspection manifest  {manifest}")
    project.inspect(manifest, offline)


def menu(project, offline):
    while True:
        heading("AgriInspect AI")
        print(f"  Context mode  {'Offline (cache only)' if offline else 'Live with cache fallback'}\n")
        print("  [1] Train and compare models\n  [2] Inspect a batch from CSV\n"
              "  [3] View latest inspection\n  [4] Check weather and market prices\n"
              "  [5] Inspect a photo or folder\n  [0] Exit")
        choice = input("\n  Select an option> ").strip()
        try:
            if choice == "0":
                return
            if choice == "1":
                project.fit(choose_path("Training CSV", project.cfg["training_csv"]))
            elif choice == "2":
                project.inspect(choose_path("Batch CSV", project.cfg["inspection_csv"]), offline)
            elif choice == "3":
                runs = sorted(project.root.glob("runs/*/summary.txt"), key=lambda p:p.stat().st_mtime_ns)
                if runs:
                    summary = runs[-1]
                    folder = summary.parent
                    context = json.loads((folder / "context.json").read_text(encoding="utf8"))
                    inspection_report(folder, summary.read_text(encoding="utf8"), context)
                    heading("Figure interpretation")
                    print((folder / "figures/interpretation.txt").read_text(encoding="utf8").strip())
                else:
                    print("Inspect a batch first.")
            elif choice == "4":
                context_report(MarketDataFetcher(project.cfg).fetch_live_market_data(offline))
            elif choice == "5":
                source = choose_path("Photo or folder path")
                print(f"  Crops  {', '.join(sorted(CROPS))}")
                crop = input("  Crop> ")
                batch = input("  Batch name (Enter to use the file or folder name)> ")
                inspect_images(project, source, crop, batch, offline)
            else:
                print("Enter a number from 0 to 5.")
        except ERRORS as exc:
            print(f"Could not complete operation: {exc}")


def main():
    parser = argparse.ArgumentParser(description="AgriInspect AI crop batch quality grader")
    parser.add_argument("--config",type=Path,default=ROOT/"config.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", type=Path, help="Train from a metadata CSV")
    mode.add_argument("--model", type=Path, help="Load a trusted model.pkl saved by this project")
    mode.add_argument("--demo", action="store_true", help="Run the bundled synthetic trial offline")
    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument("--inspect", type=Path, help="Inspection metadata CSV")
    inspection.add_argument("--inspect-images", type=Path, help="Inspect an independent photo or folder of one crop")
    parser.add_argument("--crop", help="Crop name for --inspect-images")
    parser.add_argument("--batch", help="Batch name for --inspect-images (default: file or folder name)")
    parser.add_argument("--offline",action="store_true",help="Use cached context only")
    args = parser.parse_args()
    if args.inspect_images and not args.crop:
        parser.error("--inspect-images requires --crop")
    if not args.inspect_images and (args.crop is not None or args.batch is not None):
        parser.error("--crop and --batch require --inspect-images")
    if args.demo and (args.inspect or args.inspect_images):
        parser.error("--demo runs its own inspection and cannot be combined with --inspect or --inspect-images")
    try:
        cfg = read_config(args.config)
        if args.demo:
            heading("AgriInspect AI | Offline demo")
            print("  Training and inspection use synthetic images.")
            project = Project(cfg | {"output_dir": cfg["output_dir"] / "demo"})
            project.fit(ROOT/"data/demo/train.csv")
            project.inspect(ROOT/"data/demo/inspect.csv", True)
        else:
            project = Project(cfg)
            if args.model:
                project.load_model(args.model)
                heading("Saved model loaded")
                print(f"  {args.model}")
            if args.train:
                project.fit(args.train)
            if args.inspect:
                project.inspect(args.inspect, args.offline)
            elif args.inspect_images:
                inspect_images(project, args.inspect_images, args.crop, args.batch, args.offline)
            elif not args.train:
                menu(project, args.offline)
    except ERRORS as exc:
        parser.exit(2, f"AgriInspect: {exc}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nSession ended.")
