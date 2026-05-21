# Copyright (c) OpenMMLab. All rights reserved.
"""Diffusion-Prior Feature Enhancement (DiffPrior-FPN) for Dark-DINO.

This module uses the frozen Stable Diffusion VAE encoder to extract
illumination-invariant semantic priors, which are then fused with the
dark-image features via cross-attention.  The key idea: the SD VAE has
been trained on millions of normal-light images, so its latent space
contains strong semantic priors that are agnostic to illumination.

The fusion module:
1. Projects SD VAE latents from 4 channels to the neck's channel dim.
2. Spatially aligns (interpolates) the VAE features to each FPN level.
3. Applies cross-attention: dark features (Q) attend to SD features (K/V).
4. Uses a residual gate to control the influence of the SD prior.

This is the third core contribution of Dark-DINO.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig

from ..utils.sd_vae_prior import SDVAEPrior

logger = logging.getLogger(__name__)


class CrossAttentionFusion(BaseModule):
    """Cross-attention fusion between dark features and SD VAE priors.

    Dark features serve as Query, SD VAE features as Key/Value.

    Args:
        channels (int): Feature channel dimension. Default 256.
        num_heads (int): Number of attention heads. Default 4.
        dropout (float): Dropout rate. Default 0.0.
    """

    def __init__(
        self,
        channels: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

        # Residual gate (controls how much SD prior is injected)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, dark_feat: Tensor, sd_feat: Tensor) -> Tensor:
        """Cross-attention fusion.

        Args:
            dark_feat (Tensor): Dark-image features (B, C, H, W).
            sd_feat (Tensor): SD VAE prior features (B, C, H', W').

        Returns:
            Tensor: Fused features (B, C, H, W).
        """
        B, C, H, W = dark_feat.shape
        residual = dark_feat

        # Reshape to (B, N, C)
        q = dark_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        kv = sd_feat.flatten(2).transpose(1, 2)  # (B, H'*W', C)

        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        Q = self.q_proj(q).reshape(B, -1, self.num_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3)  # (B, heads, N_q, head_dim)
        K = self.k_proj(kv).reshape(B, -1, self.num_heads, self.head_dim)
        K = K.permute(0, 2, 1, 3)
        V = self.v_proj(kv).reshape(B, -1, self.num_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, H * W, C)
        out = self.out_proj(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        # Residual gate
        return residual + self.gate * out


@MODELS.register_module()
class DiffPriorFPN(BaseModule):
    """Diffusion-Prior Feature Enhancement module.

    Applies SD VAE cross-attention fusion to each FPN level.

    Args:
        channels (int): Feature channel dimension. Default 256.
        num_levels (int): Number of FPN levels. Default 4.
        vae_model (str): SD VAE model name. Default
            'stabilityai/sd-vae-ft-mse'.
        vae_proj_channels (int): Intermediate channel dim for VAE projection.
            Default 64.
        num_heads (int): Number of cross-attention heads per level. Default 4.
        enable (bool): Whether to enable DiffPrior (config switch for ablation).
            Default True.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        channels: int = 256,
        num_levels: int = 4,
        vae_model: str = 'stabilityai/sd-vae-ft-mse',
        vae_proj_channels: int = 64,
        num_heads: int = 4,
        enable: bool = True,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.num_levels = num_levels
        self.enable = enable

        if not enable:
            return

        # SD VAE prior (lazy-loaded)
        self._sd_vae = SDVAEPrior(model_name=vae_model)

        # VAE feature projection: 4 channels → channels
        self.vae_proj = nn.Sequential(
            nn.Conv2d(4, vae_proj_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(vae_proj_channels, channels, 1, bias=True),
        )

        # Per-level cross-attention fusion
        self.fusion_layers = nn.ModuleList([
            CrossAttentionFusion(channels, num_heads)
            for _ in range(num_levels)
        ])

    def forward(self, feats: Tuple[Tensor, ...],
                batch_inputs: Tensor) -> Tuple[Tensor, ...]:
        """Apply diffusion-prior feature enhancement.

        Args:
            feats (tuple[Tensor]): Multi-level features from FreqDec-Neck,
                each (B, C, H, W).
            batch_inputs (Tensor): Original dark images (B, 3, H, W).

        Returns:
            tuple[Tensor]: Enhanced multi-level features.
        """
        if not self.enable:
            return feats

        # Extract SD VAE features
        sd_feats = self._sd_vae.encode_features(batch_inputs)
        if sd_feats is None:
            # VAE not available; return features unchanged
            return feats

        # Project VAE features to detection channel dim
        sd_feats = self.vae_proj(sd_feats)  # (B, C, H/8, W/8)

        outs = []
        for lvl, feat in enumerate(feats):
            if lvl < len(self.fusion_layers):
                # Spatially align SD features to current FPN level
                H, W = feat.shape[-2:]
                sd_aligned = F.interpolate(
                    sd_feats, size=(H, W), mode='bilinear',
                    align_corners=False)
                enhanced = self.fusion_layers[lvl](feat, sd_aligned)
                outs.append(enhanced)
            else:
                outs.append(feat)

        return tuple(outs)

    def train(self, mode: bool = True):
        """Override to keep SD VAE frozen."""
        super().train(mode)
        # SD VAE should always be in eval mode
        if self._sd_vae is not None:
            self._sd_vae.eval()
        return self
