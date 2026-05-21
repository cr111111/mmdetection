# Copyright (c) OpenMMLab. All rights reserved.
"""2D Discrete Cosine Transform (DCT) utilities for frequency-domain analysis.

This module provides a pure PyTorch implementation of 2D Type-II DCT and its
inverse (Type-III, aka IDCT), along with frequency-band slicing utilities
used by the FreqDecoupledNeck.  The implementation follows the orthogonality-
preserving formulation so that ``idct2(dct2(x)) ≈ x`` to machine precision.

Reference:
    - https://arxiv.org/abs/cmp-lin/9906003  (DCT definition)
    - M. R. Portnoff, "Short-time Fourier analysis of sampled signals"
"""

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _dct_basis(N: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build the N×N Type-II DCT basis matrix (orthogonal).

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
        x (Tensor): Input tensor of shape (..., H, W).

    Returns:
        Tensor: DCT coefficients of the same shape.
    """
    H, W = x.shape[-2], x.shape[-1]
    basis_h = _dct_basis(H, x.device, x.dtype)  # (H, H)
    basis_w = _dct_basis(W, x.device, x.dtype)  # (W, W)
    # DCT_2D = basis_h @ x @ basis_w^T
    out = basis_h @ x @ basis_w.t()
    return out


def idct2(x: Tensor) -> Tensor:
    """Apply 2D Type-III IDCT (inverse of dct2) along the last two dims.

    Args:
        x (Tensor): DCT coefficients of shape (..., H, W).

    Returns:
        Tensor: Reconstructed spatial-domain signal of the same shape.
    """
    H, W = x.shape[-2], x.shape[-1]
    basis_h = _dct_basis(H, x.device, x.dtype)  # orthogonal → inv = transpose
    basis_w = _dct_basis(W, x.device, x.dtype)
    out = basis_h.t() @ x @ basis_w
    return out


def freq_band_masks(
    h: int,
    w: int,
    low_ratio: float = 0.25,
    high_ratio: float = 0.75,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate binary masks for low / mid / high frequency bands in 2D DCT.

    The "radius" of each frequency coefficient is its normalised Euclidean
    distance from the DC component at (0, 0):

        r(u, v) = sqrt((u/H)^2 + (v/W)^2)

    Args:
        h (int): Spatial height.
        w (int): Spatial width.
        low_ratio (float): Normalised radius threshold below which is "low".
        high_ratio (float): Normalised radius threshold above which is "high".
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Three (h, w) boolean masks
            (low_mask, mid_mask, high_mask).
    """
    u = torch.arange(h, device=device, dtype=dtype).unsqueeze(1)  # (h, 1)
    v = torch.arange(w, device=device, dtype=dtype).unsqueeze(0)  # (1, w)
    radius = ((u / h) ** 2 + (v / w) ** 2).sqrt()  # (h, w)

    low_mask = radius <= low_ratio
    high_mask = radius > high_ratio
    mid_mask = ~low_mask & ~high_mask
    return low_mask, mid_mask, high_mask


def split_freq_bands(
    x: Tensor,
    low_ratio: float = 0.25,
    high_ratio: float = 0.75,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Decompose a feature map into three frequency bands via 2D DCT.

    Args:
        x (Tensor): Feature map of shape (B, C, H, W).
        low_ratio (float): Low-band radius ratio.
        high_ratio (float): High-band radius ratio.

    Returns:
        Tuple of three tensors, each (B, C, H, W):
            - low_band: low-frequency component (spatially reconstructed)
            - mid_band: mid-frequency component
            - high_band: high-frequency component
    """
    B, C, H, W = x.shape
    coeffs = dct2(x)  # (B, C, H, W)

    low_mask, mid_mask, high_mask = freq_band_masks(
        H, W, low_ratio, high_ratio, x.device, x.dtype)

    # Apply masks and inverse-DCT each band back to spatial domain
    bands = []
    for mask in (low_mask, mid_mask, high_mask):
        # Expand mask to (1, 1, H, W) for broadcasting
        masked_coeffs = coeffs * mask.unsqueeze(0).unsqueeze(0)
        band = idct2(masked_coeffs)
        bands.append(band)

    return bands[0], bands[1], bands[2]
