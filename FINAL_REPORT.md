# AgriInspect AI — implementation and evaluation

## Inspection pipeline

A manifest connects photographs to item and batch identifiers. The image reader corrects orientation, converts inputs to RGB and resizes each image. The batch processor holds up to 32 resized images at once. NumPy applies color gains and computes HSV, grayscale and a foreground mask from the median border color. The feature vector has 31 entries: three channel means, grayscale mean, foreground fraction, estimated defect fraction, circular hue spread, and eight histogram bins per channel.

The defect estimate counts dark or brown pixels within the estimated foreground. A fruit rule combines measured Brix and hue uniformity. A vegetable rule combines measured firmness and foliage condition. Their scores are recorded alongside image-only predictions. The thresholds are illustrative course settings. Real grading standards would need crop-specific calibration.

## Kaggle model comparison

The local Kaggle collection contained 12,000 readable photos across ten crops. Fresh folders supplied Grade A and Rotten folders supplied Grade C. Training excluded five carrot photos whose resized pixels matched images carrying the opposite label, then removed 229 repeated resized images. The cleaned set contained 11,766 unique images.

A seeded group split reserved 2,353 images for comparison and used 9,413 for development. Same-stem files in the same folder and source groups linked by identical resized images stayed together. Physical specimen IDs were unavailable.

| Model | Development CV macro F1 | Holdout accuracy | Holdout macro F1 |
|---|---:|---:|---:|
| RBF Support Vector Classifier | 0.867 | 88.1% (2,073/2,353) | 0.881 |
| K-Nearest Neighbors, k = 3 | 0.856 | 87.8% (2,066/2,353) | 0.878 |

Development cross-validation selected SVC. It was refitted on all 11,766 cleaned images and saved as `outputs/kaggle/models/model.pkl`. The comparison JSON, holdout predictions and excluded-label rows are in the same directory.

## Synthetic workflow check

The bundled experiment used 144 synthetic images of apples and capsicums. Deduplication preceded a seeded, grade-stratified group split. The development set contained 115 images and the holdout contained 29. Each model used a StandardScaler fitted within its training fold. Three-fold development macro F1 selected the model. The candidate definitions were fixed in advance. The selected model was then refitted on all 144 labeled images for inspection.

| Model | Development CV macro F1 | Held-out accuracy | Held-out macro F1 |
|---|---:|---:|---:|
| RBF Support Vector Classifier | 0.947 | 93.1% (27/29) | 0.928 |
| K-Nearest Neighbors, k = 3 | 0.814 | 86.2% (25/29) | 0.857 |

SVC selected Grade A for two of the nine Grade B holdout images. The independent 24-image demonstration produced nine A, seven B and eight C predictions. The report includes the item list and class-level precision, recall and F1 in the model comparison JSON. A red-channel foreground mean dominated the sample; across 18 measured items, weight and estimated defects had a Pearson coefficient of 0.06.

## Interpreting results

The proposal sets a goal above 85% accuracy on unseen produce. The synthetic holdout exercises the code. The Kaggle holdout measures performance within that collection. Use an independently sourced batch to evaluate performance on photos from a new farm or supplier, and keep all views of the same specimen in the same group. Real Grade B labels require a documented intermediate-quality standard. Lighting, naturally dark skin, shadows and busy backgrounds can change the defect estimate. A remote weather reading or published commodity quote can give context for a batch.

For smallholder suppliers, capture conditions may vary with access to cameras and lighting. Review errors by crop, variety, supplier and grade before using predictions in purchasing. Retain the image and allow human review of disputed downgrades. The engine grades visible appearance only.

## Verification scope

The packaged demo, comparative training, saved-model reload, inspection reports, three charts and automated checks ran successfully. Known-response tests exercised the weather and price parsers, local cache writes, cache reuse, mismatched markets and damaged cache files. The Kaggle training run used the local extracted image folders. The team ownership table and commands for both data paths are in README.md.
