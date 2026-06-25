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
├── eval.py                     # evaluation (incl. robustness attacks & custom dirs)
├── configs/                    # one YAML per model config
├── data/                       # split CSV is generated here (see Step 0)
├── scripts/
│   ├── generate_csv.py         # build the train/val/test split
│   ├── generate_means.py       # compute average images
│   └── generate_parameters.py  # fit the GMM
└── src/muflow/
    ├── model.py                # FastFlow model
    ├── dataset.py              # data loading & transforms
    ├── attacks.py              # content-preserving degradations (robustness)
    ├── constants.py            # paths & normalisation stats (env-configurable)
    └── gpu_utils.py            # automatic GPU selection
```

---

## Installation

Requires **Python ≥ 3.12** and a CUDA-capable GPU (CPU works but is slow).

```bash
git clone https://github.com/opontorno/MuFlow.git
cd MuFlow

# (recommended) create an environment
conda create -n muflow python=3.12 -y
conda activate muflow

# install the package and its dependencies
pip install -e .
```

---

## Configuring paths

All paths are read from `src/muflow/constants.py` and can be overridden with environment variables
— **no need to edit the source**:

| Variable                | Meaning                          | Default        |
|-------------------------|----------------------------------|----------------|
| `MUFLOW_DATA_DIR`       | Root of your datasets            | *(set this!)*  |
| `MUFLOW_WORKING_DIR`    | Repo root                        | auto-detected  |
| `MUFLOW_CHECKPOINT_DIR` | Where training runs are written  | `logs/`        |

```bash
export MUFLOW_DATA_DIR=/path/to/your/data
```

### Expected data layout

```
$MUFLOW_DATA_DIR/
├── datasets/
│   ├── ffhq/<sub>/*.png                        # real — training source
│   ├── celeba_hq/{train,val}/<sub>/*.jpg       # real — OOD real at test time
│   ├── WILD/{Closed_Set,Open_Set}/<gen>/*.png  # fake generators (test)
│   └── datasets_DFX/<gen>/*.png                # additional fake generators (test)
└── datasets_means/<mean_size>/<source>/*.png   # average images (see Step 1)

MuFlow/data/dataset_split_rand.csv              # train/val/test split (see Step 0)
```

You can point µFlow at **your own** datasets: any folder of real face images works as the training
source, and any folder of generators works as the test set — adjust the globs in
`src/muflow/dataset.py` accordingly.

---

## Training

### Step 0 — Build the split CSV

The dataset is indexed by a CSV (`data/dataset_split_rand.csv`, columns `path,split`). The script
auto-discovers every WILD generator (`Closed_Set` + `Open_Set`) and the DFX generators, caps each
fake generator to 1000 images and produces a per-class 70/15/15 split:

```bash
python scripts/generate_csv.py
```

### Step 1 — Compute average images

For training you only need the averages of your **real** dataset. Edit `INPUT_GLOB` at the top of
`scripts/generate_means.py`, then run:

```bash
# e.g. INPUT_GLOB = "$MUFLOW_DATA_DIR/datasets/ffhq/**/*.png"
python scripts/generate_means.py --name ffhq --mean-size 500 --num-images 1000
```

This writes `datasets_means/500/ffhq/*.png` (each image is the average of 500 random reals).

### Step 2 — Fit the GMM

```bash
python scripts/generate_parameters.py --model_name resnet50 --reals ffhq
```

**You can skip this step**: `main.py` runs it automatically if the parameters file is missing.

### Step 3 — Train

```bash
python main.py --config configs/resnet50.yaml --reals ffhq
```

Useful flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | model config YAML (pick one from `configs/`) | `configs/resnet50.yaml` |
| `--reals` | training real source (`ffhq`, `celeba_hq`, `ffhq+celeba_hq`) | `ffhq` |
| `--use_augs` | enable train-time `RandomAffine` (translate + scale, no rotation) | off |
| `--batch_size`, `--lr`, `--num_epochs` | optimisation | `32`, `1e-4`, `1000` |
| `--alpha` | target false-positive rate for the threshold | `0.1` |
| `--gpu_id` | force a GPU (default: auto-select the freest) | auto |
| `--wandb` | `online` / `offline` / `disabled` | `online` |
| `--wandb_entity`, `--wandb_project` | W&B destination | your default / `MuFlow` |

Each run writes a checkpoint (`best.pt`), the calibrated thresholds, predictions and a metrics
JSON to `$MUFLOW_CHECKPOINT_DIR/<run_name>/`.

> Tip: pass `--wandb disabled` to run without Weights & Biases.

---

## Testing

```bash
python eval.py --run_dir logs/<run_name>
```

All settings are loaded from the run's `run_config.yaml`. You can also evaluate on custom folders:

```bash
python eval.py --run_dir logs/<run> \
    --custom_dirs /path/reals /path/fakes --custom_labels 0 1
```

Robustness to degradations (JPEG, blur, Gaussian/salt-&-pepper noise, resize, flip, …) can be
toggled in the `attacks_configs` list inside `eval.py`.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{pontorno2026muflow,
  title     = {{\textmu}Flow: Leveraging Average Images for Improving
               Generalisation of Deepfake Faces Detectors},
  author    = {Pontorno, Orazio and Litrico, Mattia and
               Guarnera, Luca and Giuffrida, Valerio and
               Battiato, Sebastiano},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

---

## Contact

**Orazio Pontorno** — University of Catania — [orazio.pontorno@phd.unict.it](mailto:orazio.pontorno@phd.unict.it)

For questions, issues or reproducibility requests, please open a
[GitHub issue](https://github.com/opontorno/MuFlow/issues).

---

## License

See [LICENSE](LICENSE).
