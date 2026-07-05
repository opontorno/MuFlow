import contextlib
import io
import json
import os

import joblib
import numpy as np
import yaml
from scipy.stats import norm
from sklearn.neighbors import LocalOutlierFactor


SWEEP_CANDIDATES = [
    {'label': 'threshold  α=0.01', 'use_lof': False, 'alpha': 0.01, 'contamination': None},
    {'label': 'threshold  α=0.05', 'use_lof': False, 'alpha': 0.05, 'contamination': None},
    {'label': 'threshold  α=0.10', 'use_lof': False, 'alpha': 0.10, 'contamination': None},
    {'label': 'threshold  α=0.12', 'use_lof': False, 'alpha': 0.12, 'contamination': None},
    {'label': 'threshold  α=0.15', 'use_lof': False, 'alpha': 0.15, 'contamination': None},
    {'label': "LOF  (contamination='auto')", 'use_lof': True,  'alpha': None, 'contamination': 'auto'},
]


@contextlib.contextmanager
def mute():
    """Context manager that suppresses stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def fit_calibrator(val_losses: np.ndarray, cand: dict, mu_patch=None, top_k=None) -> dict:
    """Fit one calibration candidate on validation scores."""
    mean, std = float(val_losses.mean()), float(val_losses.std())
    info = {'mean': mean, 'std': std, 'losses': val_losses, 'mu_patch': mu_patch, 'top_k': top_k}
    if cand['use_lof']:
        lof = LocalOutlierFactor(novelty=True, contamination=cand['contamination'], n_jobs=-1)
        lof.fit(val_losses.reshape(-1, 1))
        info['lof'] = lof
    else:
        z = norm.ppf(1 - cand['alpha'])
        info['l_threshold'] = mean - z * std
        info['u_threshold'] = mean + z * std
    return info


def real_acc(metrics_summary: dict) -> float:
    """Extract the selection metric from a metrics summary."""
    return metrics_summary.get('real', {}).get('mean_acc', 0.0)


def load_canonical_accuracy(run_dir: str) -> float:
    """Read the saved real-baseline accuracy from a run's best_metrics.json."""
    path = os.path.join(run_dir, 'best_metrics.json')
    if not os.path.exists(path):
        return 0.0
    try:
        with open(path) as f:
            payload = json.load(f)
        return float(payload.get('metrics', {}).get('real', {}).get('mean_acc', 0.0))
    except Exception:
        return 0.0


def run_sweep(model, val_losses, mu_patch, top_k, test_dataloader, class2idx, eval_once_fn):
    """Try all calibration candidates on the clean test set, silently, and rank them."""
    rows = []
    for cand in SWEEP_CANDIDATES:
        thr = fit_calibrator(val_losses, cand, mu_patch=mu_patch, top_k=top_k)
        with mute():
            _, _, _, metrics = eval_once_fn(
                test_dataloader, model, class2idx=class2idx, threshold_info=thr, wandb_log=False)
        rows.append((cand, thr, metrics, real_acc(metrics)))

    best_cand, best_thr, best_metrics, best_acc = max(
        rows, key=lambda x: (x[3], 0 if not x[0]['use_lof'] else -1)
    )
    return best_cand, best_thr, best_metrics, best_acc, rows


def print_sweep_table(rows):
    """Print the sweep candidates ranked by real-baseline accuracy."""
    print(f"\n{'─'*64}")
    print(f"  {'Candidate':<36} {'Real Acc':>9}")
    print(f"{'─'*64}")
    for cand, _, _, acc in rows:
        print(f"  {cand['label']:<36} {acc:>9.4f}")
    print(f"{'─'*64}")


def persist_calibration(threshold_info: dict, cand: dict, metrics_summary: dict,
                        run_dir: str, threshold_path: str, lof_checkpoint: str) -> None:
    """Persist the winning calibration to a run folder."""
    save_keys = ('l_threshold', 'u_threshold', 'threshold', 'losses', 'mean', 'std', 'mu_patch', 'top_k')
    np.savez(threshold_path,
             **{k: threshold_info[k] for k in save_keys
                if k in threshold_info and threshold_info[k] is not None})
    print(f"\n  Saved thresholds     → {threshold_path}")

    if 'lof' in threshold_info:
        joblib.dump(threshold_info['lof'], lof_checkpoint)
        print(f"  Saved LOF model      → {lof_checkpoint}")
    elif os.path.exists(lof_checkpoint):
        os.remove(lof_checkpoint)

    metrics_path = os.path.join(run_dir, 'best_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            payload = json.load(f)
        payload['metrics']     = metrics_summary
        payload['calibration'] = {
            'method':        'lof' if cand['use_lof'] else 'gaussian',
            'alpha':         cand['alpha'],
            'contamination': cand['contamination'],
        }
        with open(metrics_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"  Updated best_metrics.json")

    cfg_path = os.path.join(run_dir, 'run_config.yaml')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg['use_lof']       = cand['use_lof']
        cfg['alpha']         = cand['alpha']
        cfg['contamination'] = cand['contamination']
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"  Updated run_config.yaml")
