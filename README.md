# MMA3001 Meat Quality

Estimating the severity of defects on packaged pork rasher trays from a single photo and turning that into an estimate for a pass/fail decision.

Github Project MMA3001 (Monsash University)

## What it is for

Food packaging lines need automated inspection. This tool uses a **defect area fraction** (total area of a annotated defect box as a share of the image) for a tray photo. This is then compared the number to a threshold to give a **pass/fail** decision. It is a severity proxy and screening aid. Doesn't locate defect location.

## Features

- Predicts defect area fraction and a pass/fail decision for one 720 x 540 image.
- 48 handcrafted features: HSV colour histograms and finite difference edge strength.
- Three regression models compared (linear, decision tree, SVR) against a predict the mean reference.
- Bootstrap confidence intervals, robustness tests (brightness, blur) and failure case analysis.
- Input validation: bad files are rejected with a clear error, never used to be guessed at.
- Profiled and optimised feature extraction. (about 28% faster and identical output)
- Unit tests with hand computed expected answers.

## Documentation

- API documentation (HTML): open `docs/index.html` (regenerate with `pdoc -d google meatqa -o docs`)
- Full write-up: the project report submitted on Moodle
- Experiment outputs: `results/`

## Installation

git clone https://github.com/rdas0013/mma3001-meat-quality.git
cd mma3001-meat-quality
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

Check it works:

pytest

## Data

Pork Rasher Error (Packaging), v4, from Roboflow Universe (`hello-2aqe0/pork-rasher-error-packaging`), licence CC BY 4.0. It is not stored in this repository. Download version 4 in **YOLOv8** format and extract it so that `data/train`, `data/valid`, `data/test` and `data/data.yaml` exist.

## Quick start

Build the features and the model, then assess an image:

python scripts/make_filtered_train.py data
python scripts/extract_features.py data
python scripts/train_final_model.py
python scripts/assess_cli.py path/to/tray.jpg

Example output:
{'image': 'tray.jpg', 'predicted_area_fraction': 0.0757, 'decision': 'fail', 'threshold': 0.04}

A rejected input looks like this:
REJECTED notes.md: Unsupported file type '.md'; use ['.jpeg', '.jpg', '.png']


From Python:

```python
from meatqa.assess import assess_image, load_model
bundle = load_model("models/svr_model.joblib")
print(assess_image("tray.jpg", bundle))
```

| | Description |
|---|---|
| Input | RGB image, 720 x 540 pixels, `.jpg`, `.jpeg` or `.png` |
| Output | `predicted_area_fraction` (>= 0, fraction of image area), `decision` (`pass`/`fail`), `threshold` (default 0.04) |
| Errors | `FileNotFoundError` for a missing file; `ValueError` for an unsupported type, unreadable file or wrong size |

## How it works

1. Each image becomes 48 numbers: 32 colour-histogram values (16 hue, 8 saturation, 8 value bins) and 16 edge-strength values (mean gradient magnitude on a 4 x 4 grid).
2. A scaler (fitted on training data only) standardises them.
3. An RBF support vector regression model predicts the defect area fraction.
4. Predictions are clipped at 0 and compared with a threshold chosen on the validation split.

(Features: 16 hue + 8 saturation + 8 value histogram bins, plus mean gradient magnitude on a 4 x 4 grid, which gives 16 + 8 + 8 + 16 = 48 values in the code's grid.)

## Repository layout

| Path | Contents |
|---|---|
| `src/meatqa/` | Tested library code |
| `scripts/` | Experiment scripts that write to `results/` |
| `tests/` | pytest tests |
| `results/` | Experiment outputs |
| `docs/` | Generated HTML documentation |

## Known limitations

- Images are centre cropped by the dataset's preprocessing, so the target measures visible defect extent only.
- The test set has 80 images, so every metric is noisy.
- Predictions degrade sharply under darkening and blur.

See project report for a full list.


## Licence and citation

Code: MIT, see `LICENSE`. Dataset: CC BY 4.0, credit to the Roboflow user `hello-2aqe0`.

To cite: D'Assisi, R. (2026). *MMA3001 Meat Quality* [Software]. https://github.com/rdas0013/mma3001-meat-quality