# Copyright (c) OpenMMLab. All rights reserved.
"""Retinex physics-consistency loss for Dark-DINO.

This loss enforces three physical constraints on the Retinex decomposition
I = R * L:

1. **Reconstruction loss**:  ||I - R * L||_1  — the decomposition must be
   faithful to the observed image.
2. **Illumination smoothness loss**:  ||grad(L)||_1  — the illumination map
   should be piece-wise smooth (a core Retinex assumption).
3. **Reflectance colour consistency**:  ||R_c - mean(R, dim=c)||_1  — the
   reflectance should be roughly grey-world (no colour bias from lighting).

All three terms are differentiable and can be logged individually for
ablation analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS


def _grad_x(x: Tensor) -> Tensor:
    """Horizontal gradient (finite difference)."""
    return x[:, :, :, 1:] - x[:, :, :, :-1]


def _grad_y(x: Tensor) -> Tensor:
    """Vertical gradient (finite difference)."""
    return x[:, :, 1:, :] - x[:, :, :-1, :]


@MODELS.register_module()
class RetinexConsistencyLoss(BaseModule):
    """Retinex physics-consistency loss.

    Args:
        recon_weight (float): Weight for reconstruction loss. Default 1.0.
        smooth_weight (float): Weight for illumination smoothness loss.
            Default 0.5.
        color_weight (float): Weight for reflectance colour consistency loss.
            Default 0.1.
        eps (float): Small constant for numerical stability. Default 1e-6.
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        smooth_weight: float = 0.5,
        color_weight: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.recon_weight = recon_weight
        self.smooth_weight = smooth_weight
        self.color_weight = color_weight
        self.eps = eps

    def forward(
        self,
        I: Tensor,
        R: Tensor,
        L: Tensor,
    ) -> dict:
        """Compute Retinex consistency losses.

        Args:
            I (Tensor): Original input image (B, 3, H, W).
            R (Tensor): Predicted reflectance (B, 3, H, W).
            L (Tensor): Predicted illumination (B, 3, H, W).

        Returns:
            dict: Dictionary of loss tensors:
                - loss_recon: Reconstruction loss.
                - loss_smooth: Illumination smoothness loss.
                - loss_color: Reflectance colour consistency loss.
                - loss_retinex: Weighted sum of the above.
        """
        # 1. Reconstruction: ||I - R * L||_1
        recon = F.l1_loss(I, R * L)

        # 2. Illumination smoothness: ||grad(L)||_1
        grad_x = _grad_x(L)
        grad_y = _grad_y(L)
        smooth = (grad_x.abs().mean() + grad_y.abs().mean()) / 2.0

        # 3. Reflectance colour consistency: each channel should be close to
        #    the channel mean (grey-world assumption for reflectance)
        R_mean = R.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        color = F.l1_loss(R, R_mean.expand_as(R))

        loss_retinex = (self.recon_weight * recon
                        + self.smooth_weight * smooth
                        + self.color_weight * color)

        return dict(
            loss_recon=recon * self.recon_weight,
            loss_smooth=smooth * self.smooth_weight,
            loss_color=color * self.color_weight,
            loss_retinex=loss_retinex,
        )
