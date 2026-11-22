"""Coordinate training, independent batch inspection and external context."""
import os
import pickle
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from .processor import BatchProcessor
from .ml_classifier import conflicting_labels, train
from ..abdul_haseeb.data_handler import read_metadata
from ..abdul_haseeb.api import MarketDataFetcher
from ..abdul_haseeb.utils import write_json
from ..mohammad_taha import analytics, visualizer
from ..mohammad_taha.terminal import heading, inspection_report


class Project:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg["output_dir"]
        self.model = self.last = None

    def process(self, path):
        batch = BatchProcessor(path.stem, self.cfg["image_size"], self.cfg["channel_gain"])
        try:
            frame, columns = batch.process(read_metadata(path))
        finally:
            if batch.errors:
                log = self.root / f"skipped-{uuid4().hex[:8]}.json"
                write_json(log, batch.errors)
                print(f"Skipped {len(batch.errors)} unreadable images. Details: {log}")
        return frame, columns

    def fit(self, path):
        heading("Model training")
        print(f"  Training manifest  {path}")
        print("  Reading photos and extracting image features...", flush=True)
        frame, columns = self.process(path)
        training = frame
        excluded = None
        if self.cfg.get("exclude_conflicting_labels") and "grade" in frame:
            conflicts = conflicting_labels(frame)
            path_column = "original_path" if "original_path" in frame else "image"
            excluded = frame.loc[conflicts, ["item_id", path_column, "grade", "image_hash"]]
            training = frame.loc[~conflicts]
        print(f"  Comparing SVC and KNN on {len(training):,} labeled photos before deduplication...", flush=True)
        model, metrics, heldout = train(training, columns, self.cfg["seed"])
        metrics["training_manifest"] = str(path.name)
        metrics["readable_rows"] = len(frame)
        metrics["conflicting_label_rows"] = len(frame) - len(training)
        write_json(self.root / "models/comparison.json", metrics)
        heldout.to_csv(self.root / "models/held_out_predictions.csv",index=False)
        if excluded is not None:
            exclusion_report = self.root / "models/excluded_conflicting_labels.csv"
            excluded.to_csv(exclusion_report, index=False)
            if not excluded.empty:
                print(f"  Excluded {len(excluded):,} photos with conflicting labels. See {exclusion_report}.")
        self.model, self.columns = model, columns
        self.hashes = set(frame.image_hash)
        self.groups = set(frame.source_id.dropna()) if "source_id" in frame else set()
        self.training_path = path
        self.training_seed = self.cfg["seed"]
        artifact = self.root / "models/model.pkl"
        self.save_model(artifact)
        print(f"  Images used             {metrics['unique_rows']:,} unique "
              f"({metrics['holdout_rows']:,} held out during comparison)")
        print(f"  Grades learned          {', '.join(metrics['labels'])}")
        print(f"  Selected by training CV {metrics['selected']}")
        print()
        print(f"  {'Model':<8} {'Holdout accuracy':>16} {'Macro F1':>10}")
        print(f"  {'-' * 8} {'-' * 16} {'-' * 10}")
        for key, stats in metrics["models"].items():
            held_out = stats["held_out"]
            print(f"  {key:<8} {held_out['accuracy']:>16.1%} {held_out['macro avg']['f1-score']:>10.3f}")
        print(f"\n  Saved model    {artifact}")
        print(f"  Model reports  {artifact.parent}")
        return metrics

    def save_model(self, path):
        artifact = {
            "model": self.model, "columns": self.columns, "hashes": self.hashes, "groups": self.groups,
            "training_manifest": self.training_path.name, "image_size": self.cfg["image_size"],
            "channel_gain": self.cfg["channel_gain"], "seed": self.training_seed,
        }
        stream = NamedTemporaryFile("wb", dir=path.parent, delete=False)
        temp = Path(stream.name)
        try:
            with stream:
                pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def load_model(self, path):
        """Load a trusted model previously saved by this project."""
        try:
            with path.open("rb") as stream:
                artifact = pickle.load(stream)
        except (pickle.UnpicklingError, EOFError, ImportError, AttributeError) as exc:
            raise ValueError(f"Could not read saved model {path}. Retrain with this environment.") from exc
        if any(artifact[key] != self.cfg[key] for key in ("image_size", "channel_gain")):
            raise ValueError("Saved model requires the same image_size and channel_gain as training.")
        self.model, self.columns = artifact["model"], artifact["columns"]
        self.hashes, self.groups = artifact["hashes"], artifact["groups"]
        self.training_path = Path(artifact["training_manifest"])
        self.training_seed = artifact["seed"]

    def inspect(self, path, offline):
        if self.model is None:
            self.fit(self.cfg["training_csv"])
        frame, _ = self.process(path)
        frame["image"] = frame.image.map(lambda image: os.path.relpath(image, path.resolve().parent))
        if self.hashes.intersection(frame.image_hash) or ("source_id" in frame and self.groups.intersection(frame.source_id)):
            raise ValueError("Inspection includes a training image or source. Use an independent batch.")
        frame["predicted_grade"] = self.model.predict(frame[self.columns])
        context = MarketDataFetcher(self.cfg).fetch_live_market_data(offline)
        context["location"] = {k:self.cfg[k] for k in ["latitude","longitude","price_type"]}
        run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid4().hex[:6]
        folder = self.root / "runs" / run
        summary = analytics.export(frame, context, folder)
        visualizer.plot(frame, folder / "figures", self.model.classes_)
        write_json(folder/"provenance.json", {"training_manifest":self.training_path.name, "inspection_manifest":path.name,
                   "image_size":self.cfg["image_size"], "channel_gain":self.cfg["channel_gain"],
                   "seed":self.training_seed,
                   "model":type(self.model.steps[-1][1]).__name__})
        analytics.context_history(self.root)
        self.last = folder
        inspection_report(folder, summary, context)
        return folder
