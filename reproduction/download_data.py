"""
Data acquisition for reproduction experiments.

Sources:
  - ETTh1: Electricity Transformer Temperature (already in project)
    From: Zhou et al. "Informer: Beyond Efficient Transformer for
    Long Sequence Time-Series Forecasting", AAAI 2021 Best Paper.
    https://github.com/zhouhaoyi/ETDataset

  - Synthetic industrial images: Generated for vision module demo.
    Six defect types mimicking NEU surface defect dataset classes:
    crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.
    Reference: Song et al. "A Noise Robust Method Based on Completed
    Local Binary Patterns for Hot-Rolled Steel Strip Surface Defects",
    Applied Surface Science, 2014.
"""

import os
import sys
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "reproduction" / "data"


def ensure_dirs() -> None:
    for d in [
        DATA_DIR / "etth1",
        DATA_DIR / "vision" / "normal",
        DATA_DIR / "vision" / "defect",
    ]:
        d.mkdir(parents=True, exist_ok=True)


def copy_etth1() -> None:
    """Copy ETTh1 from TimeSeriesScientist/dataset into reproduction/data."""
    src = REPO_ROOT / "TimeSeriesScientist" / "dataset" / "ETTh1.csv"
    dst = DATA_DIR / "etth1" / "ETTh1.csv"
    if not src.exists():
        print(f"[SKIP] ETTh1 not found at {src}")
        return
    shutil.copy2(src, dst)
    sz_mb = dst.stat().st_size / (1024 * 1024)
    print(f"[OK] ETTh1 copied -> {dst} ({sz_mb:.1f} MB)")


def generate_synthetic_industrial_images(n_normal: int = 30, n_defect: int = 10,
                                         size: int = 224) -> None:
    """
    Generate synthetic steel surface images for vision module.

    Normal: homogeneous gray background with slight Gaussian noise.
    Defect: 6 classes with characteristic texture anomalies.
    """
    from PIL import Image, ImageDraw

    defect_types = [
        "crazing",           # fine crack network
        "inclusion",         # dark spots
        "patches",           # irregular bright patches
        "pitted_surface",    # small dark pits
        "rolled-in_scale",   # elongated dark streaks
        "scratches",         # thin bright lines
    ]

    rng = np.random.default_rng(42)

    # --- Normal images ---
    normal_dir = DATA_DIR / "vision" / "normal"
    for i in range(n_normal):
        base = rng.integers(120, 150)
        noise = rng.normal(0, 8, (size, size)).clip(-30, 30)
        img_arr = (base + noise).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="L").convert("RGB")
        img.save(normal_dir / f"normal_{i:04d}.png")
    print(f"[OK] Generated {n_normal} normal images")

    # --- Defect images ---
    defect_dir = DATA_DIR / "vision" / "defect"
    for d_type in defect_types:
        (defect_dir / d_type).mkdir(parents=True, exist_ok=True)

    for d_type in defect_types:
        for i in range(n_defect):
            base = rng.integers(120, 150)
            noise = rng.normal(0, 8, (size, size)).clip(-30, 30)
            img_arr = (base + noise).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr, mode="L").convert("RGB")
            draw = ImageDraw.Draw(img)

            if d_type == "crazing":
                for _ in range(rng.integers(10, 40)):
                    x0, y0 = rng.integers(0, size, 2)
                    x1 = x0 + rng.integers(5, 30)
                    y1 = y0 + rng.integers(5, 30)
                    if rng.random() > 0.5:
                        draw.line([(x0, y0), (x1, y1)], fill=(200, 200, 200), width=1)

            elif d_type == "inclusion":
                for _ in range(rng.integers(3, 10)):
                    cx, cy = rng.integers(10, size - 10, 2)
                    r = rng.integers(3, 12)
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 30, 30))

            elif d_type == "patches":
                for _ in range(rng.integers(2, 6)):
                    x0, y0 = rng.integers(0, size - 40, 2)
                    draw.rectangle([x0, y0, x0 + rng.integers(20, 60), y0 + rng.integers(20, 60)],
                                   fill=(220, 220, 220))

            elif d_type == "pitted_surface":
                for _ in range(rng.integers(20, 60)):
                    cx, cy = rng.integers(5, size - 5, 2)
                    draw.ellipse([cx - 1, cy - 1, cx + 1, cy + 1], fill=(40, 40, 40))

            elif d_type == "rolled-in_scale":
                for _ in range(rng.integers(3, 8)):
                    y = rng.integers(0, size)
                    draw.line([(0, y), (size, y + rng.integers(-10, 10))],
                              fill=(50, 50, 50), width=rng.integers(2, 6))

            elif d_type == "scratches":
                for _ in range(rng.integers(1, 4)):
                    x0, y0 = rng.integers(0, size, 2)
                    angle = rng.uniform(0, np.pi)
                    length = rng.integers(30, 150)
                    x1 = x0 + int(length * np.cos(angle))
                    y1 = y0 + int(length * np.sin(angle))
                    draw.line([(x0, y0), (x1, y1)], fill=(230, 230, 230), width=2)

            img.save(defect_dir / d_type / f"{d_type}_{i:04d}.png")

    print(f"[OK] Generated {n_defect * len(defect_types)} defect images "
          f"({len(defect_types)} classes x {n_defect} each)")


def main() -> None:
    ensure_dirs()
    copy_etth1()
    generate_synthetic_industrial_images(n_normal=30, n_defect=10)

    # Print summary
    total = sum(1 for _ in DATA_DIR.rglob("*") if _.is_file())
    print(f"\n[SUMMARY] {total} files ready in {DATA_DIR}")


if __name__ == "__main__":
    main()
