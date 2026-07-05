<h1 align="center">µFlow — Leveraging Average Images for Improving Generalisation of Deepfake Faces Detectors</h1>

<p align="center">
  <img src="https://img.shields.io/badge/ECCV-2026-1a73e8.svg" alt="ECCV 2026">
  <img src="https://img.shields.io/badge/Python-3.12+-3776ab.svg" alt="Python 3.12+">
  <a href="https://opontorno.github.io/MuFlow/"><img src="https://img.shields.io/badge/Project-Page-0ea5e9.svg" alt="Project Page"></a>
</p>

<p align="center">
  <b>One-class deepfake face detector trained only on real images.</b> &nbsp;·&nbsp; <b>Accepted at ECCV 2026.</b>
</p>

<p align="center">
  <img src="docs/static/images/visual_abs.png" width="85%" alt="µFlow visual abstract">
</p>

µFlow is a one-class deepfake detector trained **only on real images**. It models the distribution
of features extracted from *average* real images with a **Gaussian Mixture Model (GMM)**, and trains
a **normalizing flow (FastFlow)** to map the features of *single* real images into that
distribution. At inference, the negative log-likelihood of an image is used directly as a fakeness
score.

> This repository documents **how to train and test the model**. For the method description,
> experiments and results, see the **[project page](https://opontorno.github.io/MuFlow/)**
> (arXiv coming soon).

---

## Repository layout

```
MuFlow/
├── main.py                     # training entry point
├── eval.py                     # evaluation (robustness attacks & custom dirs)
├── configs/                    # one YAML per model config
├── data/                       # split CSV lives here
├── scripts/                    # dataset & GMM preparation
└── src/muflow/
    ├── model.py                # FastFlow model & backbone builder
    ├── dataset.py              # data loading & transforms
    ├── patch_utils.py          # native-patch sampling & feature extraction
    ├── calibration.py          # threshold calibration & sweep
    ├── attacks.py              # content-preserving degradations
    ├── constants.py            # dataset globs & normalisation stats
    └── gpu_utils.py            # automatic GPU selection
```

---

## Installation

Requires **Python ≥ 3.12** and a CUDA-capable GPU.

```bash
git clone https://github.com/opontorno/MuFlow.git
cd MuFlow

conda create -n muflow python=3.12 -y
conda activate muflow

pip install -e .
```

---

## Configuring the datasets

Point µFlow at your data root, either by exporting `MUFLOW_DATA_DIR` or editing
`src/muflow/constants.py`:

```bash
export MUFLOW_DATA_DIR=/path/to/your/data
```

The real and fake sources are glob patterns in `src/muflow/constants.py`:

- `PATH_REAL` — real images used for training, validation and calibration.
- `PATH_REAL_OOD` — real images used as the baseline at test time (set to `None` to reuse `PATH_REAL`).
- `PATH_FAKE` — list of globs for the fake generators (test only).

Expected layout:

```
$MUFLOW_DATA_DIR/
├── datasets/
│   ├── ffhq/<sub>/*.png                        # real — training source
│   ├── celeba_hq/{train,val}/<sub>/*.jpg       # real — test-time baseline
│   ├── WILD/{Closed_Set,Open_Set}/<gen>/*.png  # fake generators
│   └── other_sources/<gen>/*.png                # fake generators
└── datasets_means/<mean_size>/<source>/*.png   # average images

MuFlow/data/dataset_split_rand.csv              # train/val/test split
```

Any folder of real faces works as the training source and any folder of generators as the test
set — adjust the globs in `constants.py` accordingly.

---

## Training

### Step 0 — Build the split CSV

The dataset is indexed by a CSV (`data/dataset_split_rand.csv`, columns `path,split`):

```bash
python scripts/generate_csv.py
```

### Step 1 — Compute average images

Compute the average images of the real source (taken from `PATH_REAL` by default):

```bash
python scripts/generate_means.py
```

Pass `--input <glob> --name <folder>` to average a different source.

### Step 2 — Fit the GMM

```bash
python scripts/generate_parameters.py --model_name clip_vitl14
```

This step is optional: `main.py` runs it automatically if the parameters file is missing.

### Step 3 — Train

```bash
python main.py --config configs/clip_vitl14.yaml
```

Each run writes a checkpoint (`best.pt`), the calibrated thresholds, predictions and a metrics
JSON to `$MUFLOW_CHECKPOINT_DIR/<run_name>/`.

---

## Testing

```bash
python eval.py --run_dir logs/<run_name>
```

Settings are loaded from the run's `run_config.yaml`. You can also evaluate on custom folders:

```bash
python eval.py --run_dir logs/<run> \
    --custom_dirs /path/reals /path/fakes --custom_labels 0 1
```

Robustness degradations (JPEG, blur, noise, resize, flip, …) can be toggled in the
`attacks_configs` list inside `eval.py`.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{pontorno2026mu,
  title={$$\backslash$mu $ Flow: Leveraging Average Images for Improving Generalisation of Deepfake Faces Detectors},
  author={Pontorno, Orazio and Litrico, Mattia and Guarnera, Luca and Giuffrida, Mario Valerio and Battiato, Sebastiano},
  journal={arXiv preprint arXiv:2606.30528},
  year={2026}
}
```

---

## Contact

**Orazio Pontorno** — University of Catania — [orazio.pontorno@phd.unict.it](mailto:orazio.pontorno@phd.unict.it)

---

## License

See [LICENSE](LICENSE).
