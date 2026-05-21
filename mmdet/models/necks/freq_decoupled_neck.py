# Copyright (c) OpenMMLab. All rights reserved.
"""Frequency-Decoupled Neck (FreqDec-Neck) for Dark-DINO.

This neck performs channel mapping (like ChannelMapper) and then decomposes
each level's feature map into three frequency bands via 2D DCT:
    - **Low band** (DC + near-DC): processed by a large-kernel ConvNeXt block
      to capture global structure — the dominant reliable cue in dark images.
    - **Mid band** (edges / contours): processed by a simplified Deformable
      Attention block that captures object-level structure.
    - **High band** (fine detail / noise): processed by a Mamba / S6 block
      that suppresses noise while preserving sparse high-frequency structure.

The three bands are fused by a brightness-driven gate: the global average
illumination (from the Retinex L component) controls the mixing weights —
darker images rely more on low + mid bands, brighter images can trust high
bands more.

This is the second core contribution of Dark-DINO.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig

from ..utils.dct_utils import dct2, idct2, freq_band_masks
from ..utils.mamba_block import MambaS6Block


# ---------------------------------------------------------------------------
# Large-kernel ConvNeXt-style block for low-band
# ---------------------------------------------------------------------------
class LargeKernelConvBlock(BaseModule):
    """Large-kernel depthwise conv block (ConvNeXt-style) for low band.

    Args:
        channels (int): Number of channels.
        kernel_size (int): Depthwise conv kernel size. Default 7.
    """

    def __init__(self, channels: int, kernel_size: int = 7,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        padding = kernel_size // 2
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size,
                                 padding=padding, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pw_conv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x + residual


# ---------------------------------------------------------------------------
# Simplified Deformable Attention block for mid-band
# ---------------------------------------------------------------------------
class DeformableAttnBlock(BaseModule):
    """Simplified deformable attention for mid-band processing.

    Uses a lightweight multi-head self-attention with sinusoidal positional
    encoding (no hardcoded spatial size) for efficiency.

    Args:
        channels (int): Number of channels.
        num_heads (int): Number of attention heads. Default 4.
    """

    def __init__(self, channels: int, num_heads: int = 4,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    @staticmethod
    def _sine_pe(h: int, w: int, channels: int,
                 device: torch.device, dtype: torch.dtype) -> Tensor:
        """Generate 2D sinusoidal positional encoding ``(1, H, W, C)``."""
        # Split channels evenly for y and x axes.
        half = channels // 2
        y_pos = torch.arange(h, device=device, dtype=dtype).unsqueeze(1)
        x_pos = torch.arange(w, device=device, dtype=dtype).unsqueeze(1)
        dim_y = torch.arange(0, half, 2, device=device, dtype=dtype)
        dim_x = torch.arange(0, half, 2, device=device, dtype=dtype)
        freq_y = 1.0 / (10000 ** (dim_y / half))
        freq_x = 1.0 / (10000 ** (dim_x / half))
        pe_y = y_pos * freq_y  # (h, half/2)
        pe_x = x_pos * freq_x  # (w, half/2)
        # Interleave sin/cos
        pe_y = torch.stack([pe_y.sin(), pe_y.cos()], dim=-1).reshape(h, half)
        pe_x = torch.stack([pe_x.sin(), pe_x.cos()], dim=-1).reshape(w, half)
        pe = torch.cat([
            pe_y.unsqueeze(1).expand(h, w, half),
            pe_x.unsqueeze(0).expand(h, w, half),
        ], dim=-1)  # (H, W, C)
        return pe.unsqueeze(0)  # (1, H, W, C)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        residual = x

        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)

        # Add sinusoidal positional encoding (no spatial size limit).
        pe = self._sine_pe(H, W, C, x.device, x.dtype)
        x = x + pe

        N = H * W
        x_flat = x.reshape(B, N, C)
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return out + residual


# ---------------------------------------------------------------------------
# Brightness-driven gate
# ---------------------------------------------------------------------------
class BrightnessGate(BaseModule):
    """Brightness-driven gate for fusing three frequency bands.

    The global average illumination (scalar per image) is fed through an MLP
    to produce per-band mixing weights.  Darker images → more weight on
    low + mid bands; brighter images → more weight on high band.

    Args:
        out_channels (int): Channel dimension of the feature maps.
        num_levels (int): Number of FPN levels.
    """

    def __init__(self, out_channels: int, num_levels: int = 4,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_levels = num_levels
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3 * num_levels),  # 3 bands × num_levels
        )

    def forward(self, feats: List[Tensor],
                light_value: Tensor) -> List[Tensor]:
        """Fuse three-band features per level with brightness-driven gates.

        Args:
            feats (list[Tensor]): List of (low, mid, high) tuples per level,
                each Tensor is (B, C, H, W). Length = num_levels.
            light_value (Tensor): (B, 1) global average illumination.

        Returns:
            list[Tensor]: Fused features per level, each (B, C, H, W).
        """
        B = light_value.size(0)
        gate_logits = self.gate_mlp(light_value)  # (B, 3 * num_levels)
        gate_weights = gate_logits.reshape(B, self.num_levels, 3)
        gate_weights = F.softmax(gate_weights, dim=2)  # (B, L, 3)

        outs = []
        for lvl in range(self.num_levels):
            low, mid, high = feats[lvl]
            w = gate_weights[:, lvl, :]  # (B, 3)
            w = w.unsqueeze(-1).unsqueeze(-1)  # (B, 3, 1, 1)
            fused = w[:, 0:1] * low + w[:, 1:2] * mid + w[:, 2:3] * high
            outs.append(fused)
        return outs


# ---------------------------------------------------------------------------
# Main FreqDecoupledNeck
# ---------------------------------------------------------------------------
@MODELS.register_module()
class FreqDecoupledNeck(BaseModule):
    """Frequency-Decoupled Neck for Dark-DINO.

    After channel mapping, each level's feature is decomposed via 2D DCT into
    low / mid / high frequency bands.  Each band is processed by a specialised
    expert module (large-kernel conv / deformable attention / Mamba).  The
    three bands are then fused by a brightness-driven gate.

    Args:
        in_channels (list[int]): Input channels per scale from backbone.
        out_channels (int): Output channel dimension. Default 256.
        kernel_size (int): Kernel size for channel mapping convs. Default 1.
        num_outs (int): Number of output feature levels. Default 4.
        low_ratio (float): DCT low-band radius ratio. Default 0.25.
        high_ratio (float): DCT high-band radius ratio. Default 0.75.
        conv_cfg: Conv config for channel mapping.
        norm_cfg: Norm config for channel mapping.
        act_cfg: Activation config for channel mapping.
        use_mamba (bool): Whether to use Mamba for high-band. If False,
            falls back to a simple 1×1 conv + ReLU. Default True.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        kernel_size: int = 1,
        num_outs: int = 4,
        low_ratio: float = 0.25,
        high_ratio: float = 0.75,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = dict(type='GN', num_groups=32),
        act_cfg: OptConfigType = None,
        use_mamba: bool = True,
        init_cfg: OptMultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform'),
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)

        self.num_outs = num_outs
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio
        self.use_mamba = use_mamba

        # Channel-mapping convs (same as ChannelMapper)
        self.convs = nn.ModuleList()
        for in_ch in in_channels:
            self.convs.append(
                ConvModule(
                    in_ch, out_channels, kernel_size,
                    padding=(kernel_size - 1) // 2,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                    act_cfg=act_cfg, bias='auto'))

        # Extra convs for additional output levels (e.g. P5, P6)
        self.extra_convs = None
        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels), num_outs):
                in_ch = in_channels[-1] if i == len(in_channels) else out_channels
                self.extra_convs.append(
                    ConvModule(
                        in_ch, out_channels, 3, stride=2, padding=1,
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                        act_cfg=act_cfg, bias='auto'))

        # Per-level band experts
        self.low_experts = nn.ModuleList([
            LargeKernelConvBlock(out_channels) for _ in range(num_outs)])
        self.mid_experts = nn.ModuleList([
            DeformableAttnBlock(out_channels) for _ in range(num_outs)])
        # High-band: reuse the standalone MambaS6Block (M4 fix)
        self.high_experts = nn.ModuleList([
            MambaS6Block(channels=out_channels) if use_mamba
            else nn.Sequential(
                nn.Linear(out_channels, out_channels), nn.ReLU())
            for _ in range(num_outs)])

        # Brightness gate
        self.brightness_gate = BrightnessGate(out_channels, num_outs)

    def _channel_map(self, inputs: Tuple[Tensor]) -> List[Tensor]:
        """Apply channel mapping (same as ChannelMapper)."""
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        if self.extra_convs is not None:
            for i, conv in enumerate(self.extra_convs):
                if i == 0:
                    outs.append(conv(inputs[-1]))
                else:
                    outs.append(conv(outs[-1]))
        return outs

    def forward(self, inputs: Tuple[Tensor],
                light_tokens: Tensor = None,
                light_map: Tensor = None) -> Tuple[Tensor]:
        """Forward.

        Args:
            inputs (tuple[Tensor]): Multi-level features from the backbone
                (after Retinex decomposition, these are from R branch).
            light_tokens (Tensor, optional): (B, T, C) illumination tokens
                from RD-Backbone. Used to compute global brightness.
            light_map (Tensor, optional): (B, 3, H, W) illumination map L
                from Retinex decomposition. Used to compute global brightness.

        Returns:
            tuple[Tensor]: Multi-level fused features, same semantics as
                ChannelMapper output.
        """
        # 1. Channel mapping
        feats = self._channel_map(inputs)

        # 2. Compute global average illumination for the gate
        if light_map is not None:
            # Per-sample mean, keeps batch dimension for proper gradient flow.
            brightness = light_map.mean(dim=(1, 2, 3), keepdim=False)  # (B,)
            brightness = brightness.unsqueeze(-1)  # (B, 1)
        elif light_tokens is not None:
            brightness = light_tokens.mean(dim=(1, 2), keepdim=False)  # (B,)
            brightness = brightness.unsqueeze(-1)  # (B, 1)
        else:
            # Fallback: average feature norm per sample (gradient-friendly).
            brightness = torch.stack(
                [f.mean(dim=(1, 2, 3)) for f in feats], dim=0
            ).mean(dim=0)  # (B,)
            brightness = brightness.unsqueeze(-1)  # (B, 1)

        # 3. DCT decomposition + band-specific processing
        band_feats = []  # list of (low, mid, high) per level
        for lvl, feat in enumerate(feats):
            B, C, H, W = feat.shape

            # DCT decomposition
            coeffs = dct2(feat)
            low_mask, mid_mask, high_mask = freq_band_masks(
                H, W, self.low_ratio, self.high_ratio,
                feat.device, feat.dtype)

            # Band-specific filtering + IDCT
            low_coeffs = coeffs * low_mask.unsqueeze(0).unsqueeze(0)
            mid_coeffs = coeffs * mid_mask.unsqueeze(0).unsqueeze(0)
            high_coeffs = coeffs * high_mask.unsqueeze(0).unsqueeze(0)

            low_band = idct2(low_coeffs)
            mid_band = idct2(mid_coeffs)
            high_band = idct2(high_coeffs)

            # Band-specific expert processing
            low_out = self.low_experts[lvl](low_band)
            mid_out = self.mid_experts[lvl](mid_band)

            # High-band: experts expect (B, N, C)
            high_flat = high_band.flatten(2).transpose(1, 2)  # (B, N, C)
            high_out = self.high_experts[lvl](high_flat)
            high_out = high_out.transpose(1, 2).reshape(B, C, H, W)

            band_feats.append((low_out, mid_out, high_out))

        # 4. Brightness-driven gate fusion
        fused = self.brightness_gate(band_feats, brightness)

        return tuple(fused)
