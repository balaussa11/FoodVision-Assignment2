# FoodVision AI — Assignment 2

**Computer Vision | Food Classification: Bread, Drinks, Fruit**

A small Streamlit demo for classifying uploaded food images using a frozen pretrained
**DINOv2 ViT-S/14** feature extractor and a trained linear classification head. It
also finds the three most visually similar training photos based on cosine similarity.

## Results

The new model was evaluated on the 60-image **validation** set from Assignment 1.
These figures should not be interpreted as performance on an independent test set.

| Model | Validation accuracy | Macro F1 |
| --- | ---: | ---: |
| Simple CNN — A1 (reported, not retrained here) | 63.3% | 0.584 |
| ResNet-18 — A1 (reported, not retrained here) | 93.3% | 0.933 |
| **DINOv2 + linear head — A2** | **100%** | **1.000** |

**Retrieval Precision@3:** 0.9944 (using original training embeddings).

### Ablation study

| Variant | Validation accuracy | Macro F1 |
| --- | ---: | ---: |
| One original training view per photo | 100% | 1.000 |
| One fixed augmented training view per photo | 100% | 1.000 |

Both variants used 270 feature vectors, the same optimizer and epoch budget.
Augmentation did not improve accuracy on this validation set. The variant with
augmentation was selected using validation loss as a tie-breaker. No synthetic
photos were used. See `artifacts/metrics.json` for details.

## Files

- `app.py` — Streamlit image upload/camera interface, class prediction, confidence,
  five-run inference latency, and similar-image gallery.
- `train.py` — DINOv2 feature extraction, linear-probe training, ablation,
  classification metrics, confusion matrix, graphs, and retrieval evaluation.
- `prepare_data.py` — unzip and structure the user's original dataset.
- `audit_dataset.py` — basic data quality checks.
- `FoodVision_Colab.ipynb` — Colab runner for `FoodVision_GitHub_Ready.zip`
  and the original private dataset ZIP; does not retrain by default.
- `artifacts/` — **actual saved A2 results**, a trained classifier head, and a
  train-embedding index; pretrained DINOv2 encoder weights are downloaded on first run.
- `baseline_A1.json` — numbers reported in the Assignment 1 report.

## Run the app (without retraining)

Tested as a Google Colab project. For a local machine:

1. Install Python and extract or clone this repository.
2. In a terminal inside the repository directory, run:

   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

3. Open the localhost URL printed by Streamlit, usually `http://localhost:8501`.

`artifacts/classifier.pt` and `artifacts/train_index.npz` are already included.
On first use the publicly available `facebook/dinov2-small` encoder downloads
from Hugging Face, so internet access is required.

**Nearest-neighbor images:** The image dataset is not redistributed publicly
because its images were collected from web searches and their reuse rights
have not been verified. To display the three similar image *photos*, copy your
own authorized dataset into the following structure:

```text
food_dataset/
  train/
    bread/    (90 images)
    drinks/   (90 images)
    fruit/    (90 images)
  val/
    bread/    (20 images)
    drinks/   (20 images)
    fruit/    (20 images)
```

The gallery uses paths from `artifacts/train_index.npz`; the original images
must retain their exact filenames and folder placement. Without the images,
classification still works, but the gallery cannot show photo thumbnails.

## Retrain / reproduce experiments

With an authorized `food_dataset/` directory present, run:

```bash
python audit_dataset.py
python train.py --data-dir food_dataset --output-dir artifacts --epochs 50 --batch-size 16
```

A GPU (such as Colab T4) is recommended. Data: 270 train + 60 validation images,
balanced across the three classes. Fixed random seed: 42. The encoder is frozen;
only the 384-to-3 linear head is optimized with AdamW. The feature extractor
uses 224 × 224 RGB images and ImageNet normalization.

To start from the original private dataset ZIP instead:

```bash
python prepare_data.py --zip food_dataset.zip
```

Then run the commands above. **Do not upload `food_dataset.zip` publicly**
unless you have redistribution permission for all images.

## Limitations and ethics

- The same 60 images were used for model selection and reported validation
  accuracy. **No independent holdout test set was used**, so 100% may be optimistic.
- A three-class classifier always predicts one of `bread`, `drinks`, or `fruit`,
  even for images that show none of them.
- The gallery's nearest neighbors provide retrieval-based evidence, not an
  attention map or causal explanation.
- The dataset is small and may not represent real-world photos or foods.
- Uploaded photos are processed within the application environment; only
  upload images you have permission to use.

## Original model and sources

- Assignment 2 course handout, Astana IT University.
- Oquab et al. (2023), [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193).
- [Official Meta DINOv2 code](https://github.com/facebookresearch/dinov2).
- [DINOv2-small checkpoint](https://huggingface.co/facebook/dinov2-small).

## Submission note

The graded assignment requests a 6–8 page PDF report, a 3–5 minute demo video,
a repository URL, and an app link or local run guide. This repository provides
the source code, trained linear classifier, metrics, and local run instructions.
