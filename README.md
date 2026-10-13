# MMA3001 Meat Quality

Estimating defect severity on packaged pork rasher trays from a single photo, using regression models on hand-crafted image features.

Individual project for MMA3001 Numerical Methods and Machine Learning.

## Problem

Food-packaging lines need automated inspection. This project predicts a **defect area fraction** (total annotated defect-box area as a share of the image) for a tray photo, then turns it into a **pass/fail** decision with a threshold. It is a proxy for severity, not a defect detector: it does not locate defects.

## Inputs and outputs

| | Description |
|---|---|
| Input | One RGB image, **720 x 540 pixels**, `.jpg`, `.jpeg` or `.png` |
| Output | `predicted_area_fraction` (float, >= 0, fraction of image area), `decision` (`"pass"` or `"fail"`), `threshold` (float, default 0.04) |
| Invalid input | Missing file raises `FileNotFoundError`. Unsupported type, unreadable file or wrong size raises `ValueError`. No number is ever guessed for a bad input. |

## Data

Pork Rasher Error (Packaging), v4, Roboflow Universe project `hello-2aqe0/pork-rasher-error-packaging`, licence **CC BY 4.0**. 3,549 images (train 3,349 / valid 120 / test 80), YOLOv8 box labels, five classes. The data is **not** committed (see `.gitignore`). Download version 4 in YOLOv8 format from Roboflow and extract it so that `data/train`, `data/valid`, `data/test` and `data/data.yaml` exist.

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

## Reproduce the results

Run from the repository root, in this order:
python scripts/summarise_dataset.py data
python scripts/make_filtered_train.py data
python scripts/summarise_targets.py data
python scripts/extract_features.py data
python scripts/score_reference.py
python scripts/fit_linear.py
python scripts/compare_models.py
python scripts/tune_models.py
python scripts/bootstrap_ci.py
python scripts/profile_features.py
python scripts/image_size_experiment.py
python scripts/time_features.py
python scripts/robustness_test.py
python scripts/failure_cases.py
python scripts/pass_fail.py
python scripts/train_final_model.py

Outputs are written to `results/` (redirect each script's output to a file, as in the committed results).

## Use
python scripts/assess_cli.py path/to/tray.jpg

Or from Python:

```python
from meatqa.assess import assess_image, load_model
bundle = load_model("models/svr_model.joblib")
print(assess_image("tray.jpg", bundle))
```

## Tests
pytest


## Repository layout

| Path | Contents |
|---|---|
| `src/meatqa/` | Tested library code: data, targets, features, metrics, assess |
| `scripts/` | Thin scripts that run experiments and write to `results/` |
| `tests/` | pytest tests |
| `results/` | Experiment outputs used in the report |
| `docs/` | Generated HTML documentation (`pdoc`) |
| `data/`, `models/` | Git-ignored; regenerate as above |

## Method summary

Features: 36 numbers per image (HSV colour histograms plus finite-difference edge strength on a 4 x 4 grid). Models: linear regression, decision tree and SVR (rbf) compared against a predict-the-training-mean reference. Settings chosen on the validation split, test used for the final check, bootstrap confidence intervals on the 80 test images.

## Limitations

- Images are centre-cropped by the dataset's preprocessing and most boxes touch an image edge, so the target measures *visible* defect extent.
- An empty label file may mean no defect, or a defect that was cropped out or never annotated.
- The test set has 80 images (12 "pass"), so all metrics are noisy.
- Wrinkle has no examples in valid or test.
- Predictions degrade sharply under darkening and blur.
- Train/test class mix differs, and leakage between splits could not be fully ruled out.

## AI use

Much of the code was drafted with Claude (Anthropic) and then run, tested and checked by the author. See the AI reflection in the project report.

## Licence

MIT, see `LICENSE`. Dataset: CC BY 4.0, credit to the Roboflow user `hello-2aqe0`.
