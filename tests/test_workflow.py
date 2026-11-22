"""Regression checks for the file, image, model and network boundaries."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock
import requests
import numpy as np
import pandas as pd
from PIL import Image
from src.abdul_haseeb.api import MarketDataFetcher
from src.abdul_haseeb.data_handler import inspection_manifest, read_metadata
from src.abdul_haseeb.import_dataset import import_images, import_manifest
from src.abdul_haseeb.utils import read_config
from src.izaan_aamir.processor import BatchProcessor
from src.izaan_aamir.integration import Project

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = read_config(ROOT/"config.json")
        self.cfg.update(cache=self.root/"cache.json",output_dir=self.root/"outputs")

    def test_provider_parse_and_separate_cache_fallback(self):
        service = MarketDataFetcher(self.cfg)
        weather = Mock(status_code=200)
        weather.json.return_value = {"current":{"time":"2025-01-01T00:00","temperature_2m":24,"relative_humidity_2m":76}}
        catalog = Mock(status_code=200)
        catalog.json.return_value = {"result":{"resources":[{"name":"wfp_food_prices_pak.csv","format":"CSV","url":"https://example.org/prices"}]}}
        prices = Mock(status_code=200)
        prices.content = b"market,commodity,pricetype,date,price,unit,currency\nKarachi,Potatoes,Retail,2025-01-01,52,KG,PKR\nKarachi,Potatoes,Retail,2025-02-01,54,KG,PKR\n"
        with patch("requests.get",side_effect=[weather,catalog,prices]):
            first = service.fetch_live_market_data()
        self.assertEqual(first["prices"]["data"]["previous_price"],52)
        with patch("requests.get",side_effect=requests.ConnectionError("offline")):
            next_result = service.fetch_live_market_data()
        self.assertEqual(next_result["weather"]["status"],"cached")
        self.assertEqual(next_result["prices"]["data"]["price"],54)
        self.cfg["market"] = "Lahore"
        self.assertEqual(service.fetch_live_market_data(True)["prices"]["status"],"unavailable")
        self.cfg["cache"].write_text("bad json")
        self.assertEqual(service.fetch_live_market_data(True)["weather"]["status"],"unavailable")

    def test_import_and_skip_bad_input(self):
        source = self.root/"freshapples"
        source.mkdir()
        Image.new("RGB",(32,32),"red").save(source/"photo.png")
        import_images(self.root,self.root/"raw",1500)
        df = read_metadata(self.root/"raw/metadata.csv")
        self.assertEqual(df.grade.iloc[0],"A")
        broken = df.copy()
        broken["image"] = str(self.root/"broken.png")
        broken["item_id"] = "broken"
        p = BatchProcessor("trial",64,[1,1,1])
        frame, cols = p.process(pd.concat([broken,df],ignore_index=True))
        self.assertEqual(len(p.errors),1)
        self.assertEqual(len(frame),1)
        self.assertTrue(np.isfinite(frame[cols]).all().all())

    def test_integrated_demo_and_leakage_guard(self):
        train_csv = ROOT / "data/demo/train.csv"
        inspect_csv = ROOT / "data/demo/inspect.csv"
        app = Project(self.cfg)
        metrics = app.fit(train_csv)
        self.assertEqual(set(metrics["models"]),{"SVC","KNN"})
        restored = Project(self.cfg)
        restored.load_model(self.cfg["output_dir"] / "models/model.pkl")
        with self.assertRaisesRegex(ValueError,"independent batch"):
            restored.inspect(train_csv, True)
        run = restored.inspect(inspect_csv, True)
        self.assertEqual(len(json.loads((run/"items.json").read_text())),24)
        self.assertEqual(len(list((run/"figures").glob("*.png"))),3)

    def test_kaggle_manifest_indexes_originals_and_groups_file_formats(self):
        source = self.root / "kaggle (12000)"
        paths = ["Fruits/RottenApple/a.jpg", "Vegetables/FreshBellpepper/photo.JPG",
                 "Vegetables/FreshBellpepper/photo.png", "Vegetables/FreshBellpepper/._ignored.jpg"]
        for relative in paths:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8)).save(path)
        out = self.root / "manifests"
        import_manifest(source, out)
        frame = read_metadata(out / "metadata.csv")
        self.assertEqual(set(frame.image), {str((source / path).resolve()) for path in paths[:3]})
        peppers = frame[frame.name == "capsicum"]
        self.assertEqual(list(peppers.grade), ["A", "A"])
        self.assertEqual(peppers.source_id.nunique(), 1)
        self.assertEqual(frame.loc[frame.name == "apple", "grade"].item(), "C")

    def test_inspection_manifest_accepts_a_photo_or_folder(self):
        source = self.root / "New batch"
        source.mkdir()
        photo = source / "bell pepper.JPG"
        Image.new("RGB", (8, 8)).save(photo)
        (source / "notes.txt").write_text("Batch notes")
        out = self.root / "inputs"
        folder_manifest = inspection_manifest(source, "Bell Pepper", None, out)
        single_manifest = inspection_manifest(photo, "capsicum", "KHI-01", out)
        self.assertNotEqual(folder_manifest, single_manifest)
        frame = read_metadata(folder_manifest)
        self.assertEqual(frame[["name", "category", "batch_code", "image"]].iloc[0].tolist(),
                         ["capsicum", "vegetable", "New batch", str(photo.resolve())])
        self.assertNotIn("grade", frame)
        self.assertEqual(read_metadata(single_manifest).batch_code.item(), "KHI-01")

    def test_conflicting_labels_are_excluded_and_reported(self):
        frame = read_metadata(ROOT / "data/demo/train.csv")
        conflict = frame.iloc[:1].assign(item_id="conflict", grade="C")
        path = self.root / "training.csv"
        pd.concat([frame, conflict]).to_csv(path, index=False)
        metrics = Project(self.cfg | {"exclude_conflicting_labels": True}).fit(path)
        self.assertEqual(metrics["conflicting_label_rows"], 2)
        self.assertEqual(metrics["unique_rows"], len(frame) - 1)
        excluded = pd.read_csv(self.cfg["output_dir"] / "models/excluded_conflicting_labels.csv")
        self.assertEqual(set(excluded.item_id), {"conflict", frame.iloc[0].item_id})
