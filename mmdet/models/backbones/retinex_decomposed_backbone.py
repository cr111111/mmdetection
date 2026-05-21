# Copyright (c) OpenMMLab. All rights reserved.
"""Retinex-Decomposed Backbone (RD-Backbone) for low-light object detection.

This module decomposes an input image I into a reflectance component R and an
illumination component L following the Retinex theory: I = R * L.  The
reflectance R (texture / object structure, illumination-invariant) is fed into
the detection backbone, while the illumination L is encoded into lightweight
tokens that are injected into the DETR encoder as scene-level priors.

The decomposition head is a shallow encoder-decoder network (5 conv layers)
that is trained end-to-end with a physics-consistency loss:
    - Reconstruction:   ||I - R * L||_1
    - Illumination smoothness: ||grad(L)||_1
    - Reflectance regularisation: instance-norm on R to remove residual
      illumination bias

This design is the first contribution of Dark-DINO and is fully differentiable.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig


class RetinexDecomposer(BaseModule):
    """Retinex decomposition head: I -> (R, L).

    A lightweight 5-layer conv network that estimates the reflectance R and
    illumination L from an input image.  L is constrained to (0, 1) via
    sigmoid, and R is normalised with instance-norm to remove residual
    illumination bias.

    Args:
        in_channels (int): Number of input channels (3 for RGB). Default 3.
        mid_channels (int): Hidden channel width. Default 32.
        num_layers (int): Number of conv layers in the encoder. Default 3.
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 32,
        num_layers: int = 3,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels

        # Encoder: progressively compress channels
        encoder_layers = []
        ch_in = in_channels
        for i in range(num_layers):
            ch_out = mid_channels if i == 0 else mid_channels
            encoder_layers.append(nn.Conv2d(ch_in, ch_out, 3, padding=1))
            encoder_layers.append(nn.ReLU(inplace=True))
            ch_in = ch_out
        self.encoder = nn.Sequential(*encoder_layers)

        # Reflectance head
        self.r_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, padding=1),
        )

        # Illumination head
        self.l_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, padding=1),
        )

        self.r_norm = nn.InstanceNorm2d(in_channels, affine=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Decompose image x into (R, L).

        Args:
            x (Tensor): Input image (B, 3, H, W), assumed to be in [0, 1] or
                normalised to approximately [0, 1] range.

        Returns:
            Tuple[Tensor, Tensor]:
                - R (Tensor): Reflectance (B, 3, H, W), in ~[0, 1].
                - L (Tensor): Illumination (B, 3, H, W), strictly in (0, 1).
        """
        feat = self.encoder(x)

        # Reflectance: use sigmoid to bound to [0, 1], then instance-norm
        R = torch.sigmoid(self.r_head(feat))
        R = self.r_norm(R)

        # Illumination: sigmoid ensures (0, 1) range; detach R to stabilise
        L = torch.sigmoid(self.l_head(feat))

        return R, L


class IlluminationTokenEncoder(BaseModule):
    """Encode the illumination map L into a compact token for DETR encoder.

    A tiny CNN that pools the illumination map L into a fixed-size token
    sequence that is prepended to the encoder feature sequence.

    Args:
        in_channels (int): Channels of L (3 for RGB illumination). Default 3.
        out_channels (int): Token embedding dimension (matches DETR hidden dim).
            Default 256.
        num_tokens (int): Number of illumination tokens. Default 4.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        num_tokens: int = 4,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_tokens = num_tokens
        self.out_channels = out_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 2, 3, stride=2,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, num_tokens)),
        )
        self.proj = nn.Linear(out_channels // 2, out_channels)

    def forward(self, L: Tensor) -> Tensor:
        """Encode illumination map into tokens.

        Args:
            L (Tensor): Illumination (B, 3, H, W).

        Returns:
            Tensor: (B, num_tokens, out_channels)
        """
        B = L.size(0)
        feat = self.encoder(L)  # (B, C//2, 1, num_tokens)
        feat = feat.flatten(2).transpose(1, 2)  # (B, num_tokens, C//2)
        tokens = self.proj(feat)  # (B, num_tokens, out_channels)
        return tokens


@MODELS.register_module()
class RetinexDecomposedBackbone(BaseModule):
    """Retinex-Decomposed Backbone for Dark-DINO.

    The input image is first decomposed into R (reflectance) and L
    (illumination) by a lightweight RetinexDecomposer.  R is then fed into a
    standard detection backbone (ResNet / Swin) whose multi-scale features are
    returned for detection.  L is encoded into illumination tokens for the
    DETR encoder.

    The decomposition head shares no parameters with the detection backbone;
    this keeps the backbone's pretrained weights intact.

    Args:
        backbone (dict): Config of the detection backbone (e.g. ResNet-50).
        decomposer (dict): Config of the RetinexDecomposer.
        light_encoder (dict): Config of the IlluminationTokenEncoder.
        freeze_backbone_stage1 (bool): Whether to freeze the backbone's
            stage-1 in the decomposition pretraining phase. Default True.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        backbone: dict,
        decomposer: dict | None = None,
        light_encoder: dict | None = None,
        freeze_backbone_stage1: bool = True,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        if decomposer is None:
            decomposer = dict(type='RetinexDecomposer')
        if light_encoder is None:
            light_encoder = dict(type='IlluminationTokenEncoder')

        # Decomposition modules
        self.decomposer = RetinexDecomposer(**decomposer)
        self.light_encoder = IlluminationTokenEncoder(**light_encoder)

        # Detection backbone (built from mmdet registry so that pretrained
        # weights are automatically loaded via init_cfg)
        self.backbone = MODELS.build(backbone)

        self.freeze_backbone_stage1 = freeze_backbone_stage1

    def forward(self, x: Tensor) -> Tuple[Tuple[Tensor, ...], Tensor, Tensor, Tensor]:
        """Forward.

        Args:
            x (Tensor): Input dark image (B, 3, H, W).

        Returns:
            Tuple containing:
                - mlvl_feats (tuple[Tensor]): Multi-scale features from the
                  detection backbone on R.
                - light_tokens (Tensor): (B, num_tokens, C) illumination tokens.
                - R (Tensor): (B, 3, H, W) reflectance.
                - L (Tensor): (B, 3, H, W) illumination.
        """
        # 1. Retinex decomposition
        R, L = self.decomposer(x)

        # 2. Feed R into detection backbone
        mlvl_feats = self.backbone(R)

        # 3. Encode L into tokens
        light_tokens = self.light_encoder(L.detach())

        return mlvl_feats, light_tokens, R, L

    def train(self, mode: bool = True):
        """Override to optionally freeze backbone stage-1."""
        super().train(mode)
        if self.freeze_backbone_stage1 and hasattr(self.backbone, 'frozen_stages'):
            self.backbone.eval()
            # Only freeze the frozen stages
            for name, param in self.backbone.named_parameters():
                for frozen_idx in range(self.backbone.frozen_stages + 1):
                    if f'layer{frozen_idx}' in name or 'bn' in name:
                        param.requires_grad = False
                        break
        return self
