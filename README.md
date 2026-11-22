# AgriInspect AI

A command-line crop inspection project for AL-2001 at FAST Karachi. Inspect a directory of crop photographs using a CSV manifest, compare SVC and KNN, and export graded items, batch summaries, three figures and market context.

## Run the offline demonstration

Extract the ZIP and open a terminal in the folder containing `main.py`. Use Python 3.10 or newer.

```bash
python -m venv .venv
```

On macOS/Linux run `source .venv/bin/activate`. On Windows PowerShell run `.venv\Scripts\Activate.ps1`. Use `python3` or `py` for the initial command if needed.

```bash
python -m pip install -r requirements.txt
python main.py --demo
```

The demo uses 144 labeled synthetic images and a separate inspection batch of 24. It needs no network after dependencies are installed and writes under `outputs/kaggle/demo/` with the default configuration.

## Team ownership

| Team member | Share | Source folder and modules |
|---|---:|---|
| Izaan Aamir | 34% | `src/izaan_aamir/` contains `models.py` (produce hierarchy), `processor.py` (batch composition), `vector_engine.py` (NumPy), `ml_classifier.py` (SVC/KNN), and `integration.py` (complete pipeline). |
| Abdul Haseeb | 33% | `src/abdul_haseeb/` contains `image_loader.py` (Pillow and image exceptions), `data_handler.py` (CSV cleaning), `api.py` (weather, prices and cache), `utils.py` (configuration/JSON), `import_dataset.py` and `demo_data.py`. |
| Mohammad Taha | 33% | `src/mohammad_taha/` contains `main.py` (CLI and validation), `analytics.py` (Pandas and reports), and `visualizer.py` (plots). Documentation is in this README and FINAL_REPORT.md. |

The root launcher delegates to Mohammad Taha's CLI. The menu calls Izaan's integration layer, which uses Abdul's data services and Mohammad's reporting modules.

## Train on the local Kaggle dataset

The default configuration uses `data/kaggle/metadata.csv`, which indexes 12,000 original photos in the extracted `Fruits_Vegetables_Dataset(12000)/Fruits` and `Vegetables` folders. It covers apples, bananas, mangoes, oranges, strawberries, bell peppers (capsicums), carrots, cucumbers, potatoes and tomatoes. Fresh maps to Grade A and Rotten maps to Grade C. Grade B requires separately labeled intermediate-quality examples.

A fresh GitHub checkout includes the synthetic demo. To train on real photos, place the extracted Kaggle folders in the project and generate the manifest with the command below.

Train from the menu with `python main.py`, choose **1**, and press Enter to accept the default manifest. The equivalent one-shot command is:

```bash
python main.py --train data/kaggle/metadata.csv
```

To rebuild the manifest after updating the extracted folders, run:

```bash
python -m src.abdul_haseeb.import_dataset --source "Fruits_Vegetables_Dataset(12000)" --manifest-only --out data/kaggle
```

The command indexes photos in place and replaces `data/kaggle/metadata.csv`. The original photos and ZIPs stay in their source folders. The manifest keeps alternate file formats with the same folder and filename stem in one evaluation group. Training deduplicates identical resized images and excludes all copies of images with conflicting grades when `exclude_conflicting_labels` is enabled. Review excluded rows in `outputs/kaggle/models/excluded_conflicting_labels.csv`. The model comparison uses an internal grouped holdout from the same Kaggle collection. After evaluation, the selected model is refitted on the full cleaned set.

Use the saved model in a new menu session with `python main.py --model outputs/kaggle/models/model.pkl`. Only open model files you trust. Load them with the same Python and scikit-learn environment used for training and the same `image_size` and `channel_gain` settings.

## Inspect your own batch

### Inspect photos directly

Choose **5** in the menu and enter a photo or folder path, the crop, and an optional batch name. For folders, the CLI uses JPG, JPEG, PNG, BMP and WebP photos immediately inside the folder. Use one crop per run and photos outside the training set.

```bash
python main.py --model outputs/kaggle/models/model.pkl --inspect-images "/path/to/apple photos" --crop apple --batch KHI-01 --offline
```

You can also pass a single image to `--inspect-images`. The batch name defaults to the file or folder name. The CLI saves the generated CSV under `outputs/kaggle/inputs/` and prints the usual inspection reports.

### Inspect from a CSV

Use one CSV row per photo. Image paths are relative to that CSV. A batch can include several crop types. Use one produce item per photo and a plain, contrasting background. Choose **2** in the menu and enter the path to a separate inspection manifest. Choose **3** to read the latest summary or **4** to fetch external context.

```csv
item_id,name,category,batch_code,image,weight_gram
K01,apple,fruit,KHI-01,images/apple01.jpg,158
K02,capsicum,vegetable,KHI-01,images/capsicum01.jpg,132
```

| Field | Meaning |
|---|---|
| item_id, name, category, batch_code, image | Required. Each item ID must be unique in its CSV. Category accepts fruit or vegetable. |
| grade | Required in training. Supply A, B or C from a reference assessment. Omit for inspection. |
| source_id | Use one ID for all photos or augmentations of the same physical item. Populate it for related images before evaluating. CSVs without IDs use exact resized-image hashes as a fallback. |
| weight_gram | Positive measured weight. Missing values use the median for the same crop and category if available. Imputed weights are flagged and excluded from the weight plot. |
| brix_sweetness, acidity_percent | Optional fruit measurements. Acidity enables a sugar/acid ratio and Brix enables the illustrative fruit rule grade. |
| firmness_index, foliage_health_score | Optional vegetable indices from 0 to 1. The rule grade needs both. |

Supply at least six independent sources for every training grade and enough representation in each fold. The model uses only pixel-derived features. Grading rules use optional physical measurements independently.

```bash
python main.py --model outputs/kaggle/models/model.pkl --inspect data/my_batch/metadata.csv --offline
python main.py --train data/kaggle/metadata.csv --inspect data/my_batch/metadata.csv
```

Set `image_size` to 64 or 128 in the configuration. Set three RGB gains to adjust channel intensity. Use `--config path/to/config.json` for other deployments. Config paths resolve relative to the config file. Using a separate output directory for real work keeps synthetic reports out of the history.

## Results and external data

The terminal prints a dated run directory. Its `items.csv` and `items.json` contain item predictions and physical rule scores. It also contains category and batch tables, a Grade C list, a summary, provenance, a context snapshot, and three interpreted figures in `figures/`. The saved model, comparison metrics and held-out predictions are under `outputs/kaggle/models/`. The package includes a synthetic example run in `example_outputs/`.

Weather uses the configured coordinates. Prices use the latest available WFP Pakistan observation for the specified market, commodity and price type. The default quote is Karachi retail potatoes, and serves as a regional benchmark for other crops. Observation dates, units and currency travel with each quote. Price publications can lag current weather. Set coordinates and market together when changing locations. Market context describes conditions at inspection time.

Successful provider responses populate `data/cache/market_weather_cache.json`. Each provider falls back independently to its matching cached data with its original date. On a first offline run the corresponding status is unavailable and inspection continues. Context history is exported for comparison; a Pearson coefficient is produced only for at least three distinct observation periods with variation. Repeated snapshots in a period are averaged. Inspect the dates, crop and benchmark before interpreting a relationship.

[Open-Meteo](https://open-meteo.com/en/docs) · [WFP Pakistan food prices via HDX](https://data.humdata.org/dataset/wfp-food-prices-for-pakistan)

## Verify the project

```bash
python -m unittest discover -s tests -v
```

Tests exercise import, corrupted images, leakage checks, model comparison, inspection exports, provider response parsing and offline recovery. See FINAL_REPORT.md for the observed demo metrics, design assumptions and limits.
