"""
generate_wild_aligned.py

Aligns ALL dataset images (real + fake) using the FFHQ-style quad-transform
and saves them to {DATA_DIR}/aligned/ preserving the original directory
structure relative to DATA_DIR.

Source roots:
    ffhq/           — real training images
    celeba_hq/      — real OOD images (train/ and val/)
    WILD/           — fake images (Closed Set, Open Set)
    datasets_DFX/   — fake images (AttGAN, GDWCT, STARGAN)

On FaceNotFoundError the original image is resized and saved unchanged so
every source path maps 1-to-1 to an output path.

Usage:
    # GPU, single process (fastest per-image, ~2h for 73k images)
    python generate_wild_aligned.py --device cuda

    # CPU, multi-process (better if GPU is busy with training)
    python generate_wild_aligned.py --device cpu --num_workers 8

    # Resume a previous interrupted run
    python generate_wild_aligned.py --device cuda --resume
"""
import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from muflow.dataset import DATA_DIR, CSV_PATH
from muflow.alignment import Aligner, FaceNotFoundError


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_images(split: str) -> list[str]:
    """
    Return image paths from the CSV filtered by split.

    Args:
        split : 'test' | 'val' | 'train' | 'all'
                'all' returns every path in the CSV regardless of split.
    """
    df = pd.read_csv(CSV_PATH)
    if split != 'all':
        df = df[df['split'] == split]
    paths = df['path'].tolist()
    # Keep only paths that actually exist on disk
    paths = [p for p in paths if os.path.isfile(p)]
    return sorted(paths)


def src_to_dst(src_path: str, out_root: str) -> str:
    """
    Preserve directory structure relative to DATA_DIR.

    Examples:
        DATA_DIR/ffhq/img/00001.jpg          → out_root/ffhq/img/00001.jpg
        DATA_DIR/celeba_hq/val/img.jpg       → out_root/celeba_hq/val/img.jpg
        DATA_DIR/WILD/Closed Set/Dall-E 3/x  → out_root/WILD/Closed Set/Dall-E 3/x
        DATA_DIR/datasets_DFX/AttGAN/x.jpg   → out_root/datasets_DFX/AttGAN/x.jpg
    """
    return os.path.join(out_root, os.path.relpath(src_path, DATA_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

_aligner: Aligner | None = None


def _init_worker(output_size: int | None, device: str):
    global _aligner
    _aligner = Aligner(mode='ffhq', output_size=output_size, device=device)


def _process_one(task: tuple) -> tuple[str, bool, str]:
    """
    Align one image and save it.  Returns (src_path, aligned_ok, error_msg).
    On FaceNotFoundError the original image is saved unchanged (fallback).
    """
    src_path, dst_path = task

    try:
        img = Image.open(src_path).convert('RGB')
    except Exception as exc:
        return src_path, False, f"open error: {exc}"

    aligned_ok = True
    try:
        img_out = _aligner.align(img)   # size determined by Aligner.output_size
    except FaceNotFoundError:
        img_out    = img               # fallback: save original unchanged
        aligned_ok = False

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    ext = Path(dst_path).suffix.lower()
    fmt = 'JPEG' if ext in ('.jpg', '.jpeg') else 'PNG'
    try:
        save_kwargs = {'format': fmt}
        if fmt == 'JPEG':
            save_kwargs['quality'] = 95
        img_out.save(dst_path, **save_kwargs)
    except Exception as exc:
        return src_path, False, f"save error: {exc}"

    return src_path, aligned_ok, ''


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Align all MuFlow dataset images to FFHQ canonical positions.')
    p.add_argument('--split', type=str, default='test',
                   choices=['test', 'val', 'train', 'all'],
                   help="CSV split to process (default: test — 10,916 images). "
                        "Use 'all' to process every image in the CSV.")
    p.add_argument('--output_size', type=int, default=0,
                   help='Output canvas side in pixels. '
                        '0 (default) = preserve original input dimensions. '
                        'Set to e.g. 256 to downsample all outputs to 256×256.')
    p.add_argument('--device', type=str, default='cuda',
                   choices=['cuda', 'cpu'],
                   help='Device for FAN detection (default cuda). '
                        'Use cpu when num_workers > 1.')
    p.add_argument('--num_workers', type=int, default=1,
                   help='Parallel worker processes (default 1). '
                        'Set > 1 only with --device cpu — CUDA is not fork-safe.')
    p.add_argument('--resume', action='store_true',
                   help='Skip images whose output file already exists.')
    p.add_argument('--out_dir', type=str,
                   default=os.path.join(DATA_DIR, 'aligned'),
                   help='Output root directory (default: DATA_DIR/aligned/).')
    return p.parse_args()


def main():
    args = parse_args()

    if args.num_workers > 1 and args.device == 'cuda':
        print("[warn] CUDA + num_workers > 1 is not fork-safe. Switching to cpu.")
        args.device = 'cpu'

    print(f"DATA_DIR     : {DATA_DIR}")
    print(f"CSV          : {CSV_PATH}")
    print(f"Split        : {args.split}")
    print(f"Output root  : {args.out_dir}")
    size_label = f"{args.output_size}×{args.output_size}" if args.output_size else "preserve input size"
    print(f"Output size  : {size_label}")
    print(f"Device       : {args.device}")
    print(f"Workers      : {args.num_workers}")
    print(f"Resume       : {args.resume}")
    print()

    # ── Collect ───────────────────────────────────────────────────────────────
    print(f"Reading CSV split '{args.split}'...")
    all_paths = collect_images(args.split)
    by_root   = defaultdict(int)
    for p in all_paths:
        root = os.path.relpath(p, DATA_DIR).split(os.sep)[0]
        by_root[root] += 1
    for root, cnt in sorted(by_root.items()):
        print(f"  {root:<20} {cnt:>7} images")
    print(f"  {'TOTAL':<20} {len(all_paths):>7} images\n")

    # --output_size 0 → None → preserve input dimensions per image
    out_size = args.output_size if args.output_size > 0 else None

    tasks, skipped = [], 0
    for src in all_paths:
        dst = src_to_dst(src, args.out_dir)
        if args.resume and os.path.exists(dst):
            skipped += 1
            continue
        tasks.append((src, dst))

    if skipped:
        print(f"Skipping {skipped} already processed images (--resume).")
    print(f"Processing {len(tasks)} images...\n")

    if not tasks:
        print("Nothing to do.")
        return

    # ── Process ───────────────────────────────────────────────────────────────
    n_ok, n_fail = 0, 0
    class_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    def _record(src: str, ok: bool):
        nonlocal n_ok, n_fail
        cls = Path(src).parent.name
        class_stats[cls][0] += 1
        if ok:
            n_ok += 1
            class_stats[cls][1] += 1
        else:
            n_fail += 1

    if args.num_workers <= 1:
        _init_worker(out_size, args.device)
        for task in tqdm(tasks, desc='Aligning', unit='img', dynamic_ncols=True):
            src, ok, _ = _process_one(task)
            _record(src, ok)
    else:
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(out_size, args.device),
        ) as pool:
            futs = {pool.submit(_process_one, t): t for t in tasks}
            with tqdm(total=len(tasks), desc='Aligning', unit='img',
                      dynamic_ncols=True) as bar:
                for fut in as_completed(futs):
                    src, ok, _ = fut.result()
                    _record(src, ok)
                    bar.update(1)

    # ── Report ────────────────────────────────────────────────────────────────
    total = n_ok + n_fail
    print(f"\n{'='*65}")
    print(f"  Total processed : {total}")
    print(f"  Aligned (ok)    : {n_ok}  ({n_ok/total*100:.1f}%)")
    print(f"  Fallback (copy) : {n_fail}  ({n_fail/total*100:.1f}%)")
    print(f"{'='*65}")
    print(f"\nPer-class alignment rate (sorted by rate):")
    print(f"  {'Class':<45} {'OK':>6} {'Tot':>6} {'Rate':>7}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*7}")
    for cls, (tot, ok) in sorted(class_stats.items(),
                                  key=lambda x: x[1][1] / max(x[1][0], 1),
                                  reverse=True):
        rate = ok / tot * 100 if tot else 0
        print(f"  {cls:<45} {ok:>6} {tot:>6} {rate:>6.1f}%")
    print(f"\nOutput saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
