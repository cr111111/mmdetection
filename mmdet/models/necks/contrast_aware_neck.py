# Copyright (c) OpenMMLab. All rights reserved.
"""Contrast-Aware Enhancement Neck (CAEM) for low-contrast object detection.

This neck is designed for low-contrast object detection scenarios. It is a
drop-in replacement for :class:`ChannelMapper`, with two extra mechanisms:

1. **Local Contrast Attention (LCA)**: For each level, a local contrast map is
   computed via a sliding-window standard deviation (using the identity
   ``Var[x] = E[x**2] - E[x]**2`` evaluated with average pooling, which is
   far cheaper than ``unfold``). Regions with low local std (typical of
   low-contrast objects) are *up-weighted* through a learnable gate, so the
   network is steered to emphasise them.

2. **Cross-Scale Contrast Squeeze-Excitation (CSC-SE)**: Per-level global
   contrast statistics (mean of local std maps) are gathered, concatenated and
   fed into a lightweight MLP that produces per-level scaling factors. This
   lets the network re-balance scales whose objects are systematically of
   lower contrast.

The forward output shape exactly matches :class:`ChannelMapper`, so no other
component needs to be modified.
"""
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig


def _local_mean_std(x: Tensor, kernel_size: int, eps: float = 1e-6
                    ) -> Tuple[Tensor, Tensor]:
    """Compute per-pixel local mean and std along spatial dims using avg pool.

    Args:
        x (Tensor): Feature map of shape (B, C, H, W).
        kernel_size (int): Local window size, must be odd.
        eps (float): Numerical stability epsilon.

    Returns:
        tuple(Tensor, Tensor): ``(local_mean, local_std)``, each of shape
        ``(B, C, H, W)``.
    """
    pad = kernel_size // 2
    # E[x] over a (k, k) window.
    local_mean = F.avg_pool2d(
        x, kernel_size=kernel_size, stride=1, padding=pad,
        count_include_pad=False)
    # E[x^2] over the same window.
    local_mean_sq = F.avg_pool2d(
        x * x, kernel_size=kernel_size, stride=1, padding=pad,
        count_include_pad=False)
    local_var = (local_mean_sq - local_mean * local_mean).clamp_min(0.0)
    local_std = torch.sqrt(local_var + eps)
    return local_mean, local_std


class LocalContrastAttention(nn.Module):
    """Per-level local contrast attention module.

    For each spatial position, a low-contrast weight is generated from the
    channel-averaged local std and used as a residual gate to enhance the
    feature at low-contrast regions.
    """

    def __init__(self,
                 channels: int,
                 kernel_size: int = 7,
                 reduction: int = 4,
                 contrast_scale: float = 1.0) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel_size must be odd.'
        self.kernel_size = kernel_size
        self.contrast_scale = contrast_scale
        hidden = max(channels // reduction, 8)
        # Project the 2-channel statistic map (mean, std) into a per-position
        # gate. We deliberately keep this branch tiny.
        self.gate = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # Learnable scalar that controls how much the residual is added.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        # Channel-averaged local mean / std → (B, 1, H, W).
        local_mean, local_std = _local_mean_std(x, self.kernel_size)
        ch_mean = local_mean.mean(dim=1, keepdim=True)
        ch_std = local_std.mean(dim=1, keepdim=True)

        # Normalise std into [0, 1] per-image using its own max, so the same
        # gate behaves consistently across batches/images. detach() because we
        # only want the std map to act as a *signal*, not as a learnable path.
        std_max = ch_std.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        ch_std_norm = (ch_std / std_max).detach()
        ch_mean_norm = (ch_mean - ch_mean.mean(dim=(2, 3), keepdim=True)
                        ).detach()

        stat = torch.cat([ch_mean_norm, ch_std_norm], dim=1)  # (B, 2, H, W)
        # Low-contrast weight: high where ch_std is small.
        low_contrast_w = self.gate(stat) * (1.0 - ch_std_norm)

        return x + self.contrast_scale * self.alpha * low_contrast_w * x


class CrossScaleContrastSE(nn.Module):
    """Cross-scale contrast squeeze-excitation.

    Aggregates per-level global contrast statistics and produces per-level
    scale factors so that scales dominated by low-contrast targets are
    enhanced.
    """

    def __init__(self, num_levels: int, hidden: int = 16) -> None:
        super().__init__()
        # Two stats per level: global mean of local std and global mean.
        in_dim = num_levels * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_levels),
            nn.Sigmoid(),
        )

    def forward(self, feats: List[Tensor], stds: List[Tensor]) -> List[Tensor]:
        # Build the per-level statistics vector: (B, num_levels * 2).
        stats = []
        for f, s in zip(feats, stds):
            stats.append(s.mean(dim=(1, 2, 3), keepdim=False))   # (B,)
            stats.append(f.mean(dim=(1, 2, 3), keepdim=False))   # (B,)
        stat_vec = torch.stack(stats, dim=1)  # (B, num_levels*2)
        # Map to a [0, 1] scale per level, centred at 0.5 → multiplier in
        # [0.5, 1.5] after the affine transform below, so the SE branch can
        # never zero out a scale.
        scale = self.mlp(stat_vec)  # (B, num_levels)
        scale = 0.5 + scale  # in [0.5, 1.5]
        outs = []
        for i, f in enumerate(feats):
            s = scale[:, i].view(-1, 1, 1, 1)
            outs.append(f * s)
        return outs


@MODELS.register_module()
class ContrastAwareNeck(BaseModule):
    """Contrast-Aware Enhancement Neck.

    A drop-in replacement for :class:`ChannelMapper` that injects two
    contrast-aware mechanisms (LCA + CSC-SE). The output shape and number of
    levels are identical to :class:`ChannelMapper`, so this neck is fully
    interchangeable in any DETR-style or single-stage detector.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int): kernel_size for the channel-mapping conv. Default 3.
        conv_cfg (ConfigType, optional): Conv config of the mapping convs.
        norm_cfg (ConfigType, optional): Norm config of the mapping convs.
            Default ``dict(type='GN', num_groups=32)`` for stability.
        act_cfg (ConfigType, optional): Activation config. Default ReLU.
        bias (bool | str): Bias mode passed to ConvModule.
        num_outs (int, optional): Number of output feature maps. If larger
            than ``len(in_channels)`` extra strided convs are appended (same
            behaviour as ChannelMapper).
        local_contrast_kernel (int): Window size for local std computation.
            Default 7.
        contrast_scale (float): Strength of the LCA residual gate. Default 1.0.
        use_cross_scale_se (bool): Whether to enable the CSC-SE branch.
            Default True.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        kernel_size: int = 3,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = dict(type='GN', num_groups=32),
        act_cfg: OptConfigType = dict(type='ReLU'),
        bias: Union[bool, str] = 'auto',
        num_outs: int = None,
        local_contrast_kernel: int = 7,
        contrast_scale: float = 1.0,
        use_cross_scale_se: bool = True,
        init_cfg: OptMultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform'),
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)

        if num_outs is None:
            num_outs = len(in_channels)
        self.num_outs = num_outs
        self.local_contrast_kernel = local_contrast_kernel
        self.use_cross_scale_se = use_cross_scale_se

        # Channel-mapping convs (same as ChannelMapper).
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                ConvModule(
                    in_channel,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=bias))

        # Extra convs for additional output levels (e.g. P6).
        self.extra_convs = None
        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels), num_outs):
                if i == len(in_channels):
                    in_channel = in_channels[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    ConvModule(
                        in_channel,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        bias=bias))

        # One LCA per output scale.
        self.lca_modules = nn.ModuleList([
            LocalContrastAttention(
                channels=out_channels,
                kernel_size=local_contrast_kernel,
                contrast_scale=contrast_scale)
            for _ in range(num_outs)
        ])

        if use_cross_scale_se:
            self.cross_scale_se = CrossScaleContrastSE(num_levels=num_outs)
        else:
            self.cross_scale_se = None

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward.

        Args:
            inputs (tuple[Tensor]): Multi-level features from the backbone.

        Returns:
            tuple[Tensor]: Enhanced multi-level features. Same shape as
            :class:`ChannelMapper`.
        """
        assert len(inputs) == len(self.convs)

        # 1) Channel-map each input level.
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]

        # 2) Append extra strided levels if requested.
        if self.extra_convs is not None:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[-1]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))

        # 3) Local-contrast attention per level.
        outs = [self.lca_modules[i](outs[i]) for i in range(len(outs))]

        # 4) Optional cross-scale contrast SE.
        if self.cross_scale_se is not None:
            # Re-compute per-level local std for the SE input. Cheap because
            # convs are already done.
            stds = []
            for f in outs:
                _, s = _local_mean_std(f, self.local_contrast_kernel)
                stds.append(s.detach())
            outs = self.cross_scale_se(outs, stds)

        return tuple(outs)
