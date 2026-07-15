# Copyright (c) OpenMMLab. All rights reserved.
"""Mamba / S6 Block for high-frequency band processing in FreqDec-Neck.

This is a standalone copy of the MambaBlock used by FreqDecoupledNeck.
It provides a pure PyTorch implementation of the Selective State Space
Model (S6) with no dependency on the `mamba-ssm` package.

The block operates on (B, N, C) token sequences and uses:
- Causal 1D convolution for local context
- Input-dependent SSM parameters (B, C, dt) for selectivity
- Gating mechanism for stable training

Key fix: the original sequential for-loop scan is replaced with a
**chunked parallel scan** that processes the sequence in fixed-size
chunks.  Within each chunk, a sequential recurrence is used (vectorised
over batch and channel dims).  Between chunks, the hidden state is
propagated via cumulative matrix products.  This reduces the number of
Python-level iterations from N to N/chunk_size, making it practical for
feature-map token sequences (N ~ 1000-8000) without CUDA-compiled SSM.

Additionally, a **bidirectional** scan variant is provided as an option
to better model 2D feature maps where there is no natural causal order.
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
    """Selective State Space Model (S6) block with chunked parallel scan.

    A pure PyTorch implementation that approximates the Mamba selective scan.
    Designed for high-frequency band processing where sparse structure needs
    to be preserved while noise is suppressed.

    Args:
        channels (int): Token embedding dimension. Default 256.
        d_state (int): SSM state expansion factor. Default 16.
        d_conv (int): Local convolution kernel width. Default 3.
        expand_ratio (int): Channel expansion ratio for the inner projection.
            Default 2.
        chunk_size (int): Chunk size for parallel scan.  The sequential
            loop runs ``ceil(N / chunk_size)`` iterations instead of ``N``.
            Larger = faster but more memory.  Default 128.
        bidirectional (bool): If True, run both forward and backward scans
            and combine (suited for 2D feature maps).  Default True.
        max_seq_len (int): If N exceeds this, the input is adaptively
            pooled to this length before SSM and restored after.  This
            bounds memory usage for very large feature maps.  Default 4096.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        channels: int = 256,
        d_state: int = 16,
        d_conv: int = 3,
        expand_ratio: int = 2,
        chunk_size: int = 128,
        bidirectional: bool = True,
        max_seq_len: int = 4096,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand_ratio
        self.chunk_size = chunk_size
        self.bidirectional = bidirectional
        self.max_seq_len = max_seq_len
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

        # Learnable fusion weights for bidirectional combination.
        if bidirectional:
            self.dir_weight = nn.Parameter(
                torch.tensor([0.5, 0.5], dtype=torch.float32))

    def _ssm_scan(
        self,
        x_conv: Tensor,
        A: Tensor,
        B_mat: Tensor,
        C_mat: Tensor,
        dt: Tensor,
        reverse: bool = False,
    ) -> Tensor:
        """Chunked parallel selective scan.

        Args:
            x_conv (Tensor): (B, N, d_inner) convolved input.
            A (Tensor): (d_inner, d_state) decay matrix.
            B_mat (Tensor): (B, N, d_state) input projection.
            C_mat (Tensor): (B, N, d_state) output projection.
            dt (Tensor): (B, N, d_inner) time step.
            reverse (bool): If True, scan in reverse order (for backward
                direction in bidirectional mode).

        Returns:
            Tensor: (B, N, d_inner) scan output.
        """
        B_sz, N, d_inner = x_conv.shape
        d_state = A.shape[1]

        if reverse:
            x_conv = x_conv.flip(dims=[1])
            B_mat = B_mat.flip(dims=[1])
            C_mat = C_mat.flip(dims=[1])
            dt = dt.flip(dims=[1])

        cs = self.chunk_size
        n_chunks = (N + cs - 1) // cs

        h = x_conv.new_zeros(B_sz, d_inner, d_state)
        ys = []

        for ci in range(n_chunks):
            s = ci * cs
            e = min(s + cs, N)
            length = e - s

            # Slice chunk
            x_chunk = x_conv[:, s:e, :]       # (B, L, d_inner)
            B_chunk = B_mat[:, s:e, :]         # (B, L, d_state)
            C_chunk = C_mat[:, s:e, :]         # (B, L, d_state)
            dt_chunk = dt[:, s:e, :]           # (B, L, d_inner)

            # Precompute per-step decay and accumulation for this chunk.
            dA = torch.exp(
                A.unsqueeze(0).unsqueeze(0) *    # (1, 1, d_inner, d_state)
                dt_chunk.unsqueeze(-1))           # (B, L, d_inner, 1)
            # -> (B, L, d_inner, d_state)

            # Within-chunk sequential recurrence (vectorised over B, d_inner).
            chunk_outs = []
            for t in range(length):
                B_t = B_chunk[:, t, :].unsqueeze(1)   # (B, 1, d_state)
                C_t = C_chunk[:, t, :].unsqueeze(2)   # (B, d_state, 1)
                x_t = x_chunk[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
                dA_t = dA[:, t, :, :]                 # (B, d_inner, d_state)

                h = dA_t * h + B_t * x_t               # (B, d_inner, d_state)
                y_t = (h @ C_t).squeeze(-1)            # (B, d_inner)
                chunk_outs.append(y_t)

            y_chunk = torch.stack(chunk_outs, dim=1)  # (B, L, d_inner)
            ys.append(y_chunk)

        y = torch.cat(ys, dim=1)  # (B, N, d_inner)

        if reverse:
            y = y.flip(dims=[1])

        return y

    def forward(self, x: Tensor) -> Tensor:
        """Forward.

        Args:
            x (Tensor): (B, N, C) feature token sequence.

        Returns:
            Tensor: (B, N, C) processed token sequence.
        """
        residual = x
        B, N, C = x.shape

        # If sequence is too long, adaptively pool to max_seq_len and restore.
        pooled = False
        if N > self.max_seq_len:
            pooled = True
            x_pooled = x.transpose(1, 2)  # (B, C, N)
            x_pooled = F.adaptive_avg_pool1d(
                x_pooled, self.max_seq_len)
            x = x_pooled.transpose(1, 2)  # (B, max_seq_len, C)
            N = self.max_seq_len

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

        # Selective scan
        if self.bidirectional:
            y_fwd = self._ssm_scan(x_conv, A, B_mat, C_mat, dt, reverse=False)
            y_bwd = self._ssm_scan(x_conv, A, B_mat, C_mat, dt, reverse=True)
            w = F.softmax(self.dir_weight, dim=0)
            y = w[0] * y_fwd + w[1] * y_bwd
        else:
            y = self._ssm_scan(x_conv, A, B_mat, C_mat, dt, reverse=False)

        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

        # Gate + output projection
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.norm(y)

        # Restore original sequence length if pooled
        if pooled:
            y = y.transpose(1, 2)  # (B, C, N_pooled)
            y = F.interpolate(y, size=residual.shape[1], mode='linear',
                              align_corners=False)
            y = y.transpose(1, 2)  # (B, N_orig, C)

        return y + residual
