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

from typing import Dict

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
        recon_weight (float): Weight for reconstruction loss. Defaults to 1.0.
        smooth_weight (float): Weight for illumination smoothness loss.
            Defaults to 0.5.
        color_weight (float): Weight for reflectance colour consistency loss.
            Defaults to 0.1.
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        smooth_weight: float = 0.5,
        color_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.recon_weight = float(recon_weight)
        self.smooth_weight = float(smooth_weight)
        self.color_weight = float(color_weight)

    def forward(
        self,
        I: Tensor,
        R: Tensor,
        L: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute Retinex consistency losses.

        Each returned entry is already multiplied by its weight, so mmengine's
        loss aggregation can sum them directly without double counting.

        Args:
            I (Tensor): Original input image ``(B, 3, H, W)`` in ``[0, 1]``.
            R (Tensor): Predicted reflectance ``(B, 3, H, W)``.
            L (Tensor): Predicted illumination ``(B, 3, H, W)``.

        Returns:
            Dict[str, Tensor]: Dictionary with three weighted loss tensors

            - ``loss_recon``: weighted reconstruction loss.
            - ``loss_smooth``: weighted illumination smoothness loss.
            - ``loss_color``: weighted reflectance colour consistency loss.
        """
        # 1. Reconstruction: ||I - R * L||_1
        recon = F.l1_loss(I, R * L)

        # 2. Illumination smoothness: ||grad(L)||_1
        grad_x = _grad_x(L)
        grad_y = _grad_y(L)
        smooth = (grad_x.abs().mean() + grad_y.abs().mean()) / 2.0

        # 3. Reflectance colour consistency: each channel should be close to
        #    the channel mean (grey-world assumption for reflectance).
        R_mean = R.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        color = F.l1_loss(R, R_mean.expand_as(R))

        # Return per-term weighted losses; mmengine will sum them.
        # We do NOT also return the aggregate to avoid double counting.
        return dict(
            loss_recon=self.recon_weight * recon,
            loss_smooth=self.smooth_weight * smooth,
            loss_color=self.color_weight * color,
        )
