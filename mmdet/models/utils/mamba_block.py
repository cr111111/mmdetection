# Copyright (c) OpenMMLab. All rights reserved.
"""Mamba / S6 Block for high-frequency band processing in FreqDec-Neck.

This is a standalone copy of the MambaBlock used by FreqDecoupledNeck.
It provides a pure PyTorch implementation of the Selective State Space
Model (S6) with no dependency on the `mamba-ssm` package.

The block operates on (B, N, C) token sequences and uses:
- Causal 1D convolution for local context
- Input-dependent SSM parameters (B, C, Δ) for selectivity
- Gating mechanism for stable training
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class MambaS6Block(BaseModule):
    """Selective State Space Model (S6) block.

    A pure PyTorch implementation that approximates the Mamba selective scan.
    Designed for high-frequency band processing where sparse structure needs
    to be preserved while noise is suppressed.

    Args:
        channels (int): Token embedding dimension. Default 256.
        d_state (int): SSM state expansion factor. Default 16.
        d_conv (int): Local convolution kernel width. Default 3.
        expand_ratio (int): Channel expansion ratio for the inner projection.
            Default 2.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        channels: int = 256,
        d_state: int = 16,
        d_conv: int = 3,
        expand_ratio: int = 2,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand_ratio
        d_inner = int(channels * expand_ratio)

        self.in_proj = nn.Linear(channels, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True)

        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state * 2, d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float()
                      .unsqueeze(0).expand(d_inner, -1).clone()))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, channels, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward.

        Args:
            x (Tensor): (B, N, C) feature token sequence.

        Returns:
            Tensor: (B, N, C) processed token sequence.
        """
        residual = x
        B, N, C = x.shape

        xz = self.in_proj(x)  # (B, N, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)

        # Causal conv
        x_conv = x_branch.transpose(1, 2)  # (B, d_inner, N)
        x_conv = self.conv1d(x_conv)[:, :, :N]  # truncate for causality
        x_conv = x_conv.transpose(1, 2)  # (B, N, d_inner)
        x_conv = F.silu(x_conv)

        # SSM parameters
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        BC = self.x_proj(x_conv)  # (B, N, 2*d_state)
        B_mat, C_mat = BC.chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(BC))  # (B, N, d_inner)

        # Selective scan (sequential recurrence)
        h = x.new_zeros(B, A.shape[0], A.shape[1])  # (B, d_inner, d_state)
        ys = []
        d_inner = A.shape[0]
        for t in range(N):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            dA = torch.exp(A.unsqueeze(0) * dt_t)  # (B, d_inner, d_state)
            B_t = B_mat[:, t, :].unsqueeze(1)  # (B, 1, d_state)
            C_t = C_mat[:, t, :].unsqueeze(2)  # (B, d_state, 1)
            x_t = x_conv[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            h = dA * h + B_t * x_t
            y_t = (h @ C_t).squeeze(-1)  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, N, d_inner)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

        # Gate + output projection
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.norm(y)

        return y + residual
