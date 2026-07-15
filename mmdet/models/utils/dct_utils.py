# Copyright (c) OpenMMLab. All rights reserved.
"""2D Discrete Cosine Transform (DCT) utilities for frequency-domain analysis.

This module provides a pure PyTorch implementation of 2D Type-II DCT and its
inverse (Type-III, aka IDCT), along with frequency-band slicing utilities
used by the FreqDecoupledNeck.  The implementation follows the orthogonality-
preserving formulation so that ``idct2(dct2(x)) ~ x`` to machine precision.

Key fix: band masks use smooth (soft) transitions instead of hard binary
thresholds to avoid ringing artifacts in the reconstructed spatial bands.
"""

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _dct_basis(N: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build the N x N Type-II DCT basis matrix (orthogonal).

    Args:
        N (int): Dimension size.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Tensor: (N, N) orthogonal DCT basis.
    """
    n = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)  # (N, 1)
    k = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # (1, N)
    basis = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))  # (N, N)
    # Orthogonal scaling: C_k = sqrt(2/N) for k>0, sqrt(1/N) for k==0
    scale = torch.full((N,), math.sqrt(2.0 / N), device=device, dtype=dtype)
    scale[0] = math.sqrt(1.0 / N)
    basis = basis * scale.unsqueeze(0)
    return basis


def dct2(x: Tensor) -> Tensor:
    """Apply 2D Type-II DCT along the last two dimensions.

    Args:
        x (Tensor): Input tensor of shape ``(B, C, H, W)`` or
            ``(..., H, W)``.

    Returns:
        Tensor: DCT coefficients of the same shape.
    """
    H, W = x.shape[-2], x.shape[-1]
    basis_h = _dct_basis(H, x.device, x.dtype)  # (H, H)
    basis_w = _dct_basis(W, x.device, x.dtype)  # (W, W)
    # Flatten leading dims so matmul broadcasts correctly.
    leading = x.shape[:-2]  # (B, C, ...)
    x_flat = x.reshape(-1, H, W)  # (N, H, W)
    # DCT_2D = basis_h @ x @ basis_w^T  (per-matrix)
    out = basis_h @ x_flat @ basis_w.t()  # (N, H, W)
    return out.reshape(*leading, H, W)


def idct2(x: Tensor) -> Tensor:
    """Apply 2D Type-III IDCT (inverse of dct2) along the last two dims.

    Args:
        x (Tensor): DCT coefficients of shape ``(B, C, H, W)`` or
            ``(..., H, W)``.

    Returns:
        Tensor: Reconstructed spatial-domain signal of the same shape.
    """
    H, W = x.shape[-2], x.shape[-1]
    basis_h = _dct_basis(H, x.device, x.dtype)  # orthogonal -> inv = transpose
    basis_w = _dct_basis(W, x.device, x.dtype)
    leading = x.shape[:-2]
    x_flat = x.reshape(-1, H, W)
    out = basis_h.t() @ x_flat @ basis_w  # (N, H, W)
    return out.reshape(*leading, H, W)


def freq_band_masks(
    h: int,
    w: int,
    low_ratio: float = 0.25,
    high_ratio: float = 0.75,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
    smooth_width: float = 0.05,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate soft (smooth) masks for low / mid / high frequency bands in
    2D DCT.

    The "radius" of each frequency coefficient is its normalised Euclidean
    distance from the DC component at (0, 0)::

        r(u, v) = sqrt((u/H)^2 + (v/W)^2)

    Instead of hard binary thresholds, the masks use sigmoid transitions
    of width ``smooth_width`` around ``low_ratio`` and ``high_ratio``.
    This avoids ringing artifacts caused by sharp spectral cutoffs.

    The three masks sum to 1 everywhere (partition of unity), so
    ``low + mid + high = full spectrum``.

    Args:
        h (int): Spatial height.
        w (int): Spatial width.
        low_ratio (float): Normalised radius threshold below which is "low".
        high_ratio (float): Normalised radius threshold above which is "high".
        device: Torch device.
        dtype: Torch dtype.
        smooth_width (float): Width of the sigmoid transition band.
            Smaller = sharper cutoff (approaches binary mask).
            Larger = smoother but more spectral leakage between bands.
            Default 0.05 (about 5% of the normalised radius range).

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Three ``(h, w)`` float masks
            (low_mask, mid_mask, high_mask), each in ``[0, 1]``.
            They sum to 1 at every position.
    """
    u = torch.arange(h, device=device, dtype=dtype).unsqueeze(1)  # (h, 1)
    v = torch.arange(w, device=device, dtype=dtype).unsqueeze(0)  # (1, w)
    radius = ((u / h) ** 2 + (v / w) ** 2).sqrt()  # (h, w)

    # Sigmoid-based soft masks.
    # low_mask: 1 for r << low_ratio, 0 for r >> low_ratio
    # high_mask: 0 for r << high_ratio, 1 for r >> high_ratio
    # mid_mask = 1 - low_mask - high_mask  (partition of unity)
    s = max(smooth_width, 1e-4)

    low_mask = torch.sigmoid((low_ratio - radius) / s)
    high_mask = torch.sigmoid((radius - high_ratio) / s)
    mid_mask = 1.0 - low_mask - high_mask

    # Clamp to [0, 1] for numerical safety.
    low_mask = low_mask.clamp(0.0, 1.0)
    mid_mask = mid_mask.clamp(0.0, 1.0)
    high_mask = high_mask.clamp(0.0, 1.0)

    return low_mask, mid_mask, high_mask


def split_freq_bands(
    x: Tensor,
    low_ratio: float = 0.25,
    high_ratio: float = 0.75,
    smooth_width: float = 0.05,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Decompose a feature map into three frequency bands via 2D DCT.

    Args:
        x (Tensor): Feature map of shape (B, C, H, W).
        low_ratio (float): Low-band radius ratio.
        high_ratio (float): High-band radius ratio.
        smooth_width (float): Sigmoid transition width for soft masks.

    Returns:
        Tuple of three tensors, each (B, C, H, W):
            - low_band: low-frequency component (spatially reconstructed)
            - mid_band: mid-frequency component
            - high_band: high-frequency component
    """
    B, C, H, W = x.shape
    coeffs = dct2(x)  # (B, C, H, W)

    low_mask, mid_mask, high_mask = freq_band_masks(
        H, W, low_ratio, high_ratio, x.device, x.dtype,
        smooth_width=smooth_width)

    # Apply masks and inverse-DCT each band back to spatial domain
    bands = []
    for mask in (low_mask, mid_mask, high_mask):
        # Expand mask to (1, 1, H, W) for broadcasting
        masked_coeffs = coeffs * mask.unsqueeze(0).unsqueeze(0)
        band = idct2(masked_coeffs)
        bands.append(band)

    return bands[0], bands[1], bands[2]
