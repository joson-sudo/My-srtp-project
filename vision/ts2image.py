from __future__ import annotations

import numpy as np
from typing import Literal


def _normalize_series(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def gramian_angular_field(x: np.ndarray, method: Literal["sum", "diff"] = "sum") -> np.ndarray:
    """Convert 1D time series to 2D Gramian Angular Field image.

    GASF (sum) encodes temporal correlation via cos(arccos(x_i) + arccos(x_j)).
    GADF (diff) encodes via sin(arccos(x_i) - arccos(x_j)).

    Args:
        x: (L,) 1D time series.
        method: "sum" for GASF, "diff" for GADF.

    Returns:
        (L, L) image array in [0, 1].
    """
    x = _normalize_series(x)
    # Clip to avoid numerical issues with arccos
    x = np.clip(x, -1.0 + 1e-8, 1.0 - 1e-8)
    phi = np.arccos(x)

    if method == "sum":
        gaf = np.cos(np.add.outer(phi, phi))
    else:
        gaf = np.sin(np.subtract.outer(phi, phi))

    return (gaf + 1.0) / 2.0  # scale to [0, 1]


def markov_transition_field(x: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Convert 1D time series to 2D Markov Transition Field image.

    Quantizes the series into Q bins, then estimates the Markov transition
    matrix, and arranges transition probabilities along the temporal axis.

    Args:
        x: (L,) 1D time series.
        n_bins: number of quantile bins.

    Returns:
        (L, L) image array in [0, 1].
    """
    L = len(x)
    if L < 2:
        return np.zeros((L, L))

    bins = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    q = np.digitize(x, bins) - 1  # (L,)

    # Transition matrix
    T = np.zeros((n_bins, n_bins))
    for t in range(L - 1):
        T[q[t], q[t + 1]] += 1
    T_sum = T.sum(axis=1, keepdims=True)
    T_sum[T_sum == 0] = 1
    T = T / T_sum  # row-normalized

    # Build MTF field
    mtf = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            mtf[i, j] = T[q[i], q[j]]

    return mtf


def recurrence_plot(x: np.ndarray, epsilon: float | None = None, metric: str = "euclidean") -> np.ndarray:
    """Convert 1D time series to Recurrence Plot.

    Args:
        x: (L,) 1D time series.
        epsilon: threshold distance. If None, uses 10% of max phase-space distance.
        metric: distance metric.

    Returns:
        (L, L) binary recurrence image.
    """
    L = len(x)
    if L < 2:
        return np.zeros((L, L))

    x = _normalize_series(x)

    # Embedding into phase space with delay embedding (dim=1, delay=1)
    dist = np.abs(np.subtract.outer(x, x))

    if epsilon is None:
        epsilon = 0.1 * dist.max()

    return (dist <= epsilon).astype(np.float32)


def series_to_rgb(x: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Convert 1D time series into a 3-channel RGB image for pretrained CNNs.

    Channels:
      R: Gramian Angular Summation Field
      G: Gramian Angular Difference Field
      B: Markov Transition Field

    Args:
        x: (L,) 1D time series.
        image_size: output image size (square).

    Returns:
        (3, image_size, image_size) float32 array in [0, 1].
    """
    from PIL import Image

    gasf = gramian_angular_field(x, "sum")
    gadf = gramian_angular_field(x, "diff")
    mtf = markov_transition_field(x)

    def _resize(arr: np.ndarray) -> np.ndarray:
        img = Image.fromarray((arr * 255).astype(np.uint8))
        img = img.resize((image_size, image_size), Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0

    return np.stack([
        _resize(gasf),
        _resize(gadf),
        _resize(mtf),
    ], axis=0)  # (3, H, W)
