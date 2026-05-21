# Copyright (c) OpenMMLab. All rights reserved.
"""Retinex-Decomposed Backbone (RD-Backbone) for low-light object detection.

This module decomposes an input image I into a reflectance component R and an
illumination component L following the Retinex theory: ``I = R * L``.  The
reflectance R (texture / object structure, illumination-invariant) is fed
into the detection backbone, while the illumination L is encoded into
lightweight tokens that are injected into the DETR encoder as scene-level
priors.

The decomposition head is a shallow encoder--decoder network (3 conv layers
by default) that is trained end-to-end with a physics-consistency loss:

* Reconstruction:           ``||I - R * L||_1``
* Illumination smoothness:  ``||grad(L)||_1``
* Reflectance regularisation: grey-world prior on R

Both R and L are constrained to ``[0, 1]`` via sigmoid so that the product
``R * L`` lives in ``[0, 1]`` and matches the input image domain.

This is the first contribution of Dark-DINO and is fully differentiable.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig


class RetinexDecomposer(BaseModule):
    """Retinex decomposition head: ``I -> (R, L)``.

    A lightweight conv network that estimates the reflectance R and the
    illumination L from an input image.  Both maps are bounded in ``[0, 1]``
    via sigmoid so that their product is well-defined in the image domain.
    The reflectance head additionally uses a learnable channel-wise affine
    correction (instead of InstanceNorm, which would destroy the ``[0, 1]``
    bound) to remove residual illumination bias.

    Args:
        in_channels (int): Number of input channels (3 for RGB). Defaults to 3.
        mid_channels (int): Hidden channel width. Defaults to 32.
        num_layers (int): Number of conv layers in the shared encoder.
            Defaults to 3.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 32,
        num_layers: int = 3,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert num_layers >= 1, '`num_layers` must be >= 1'
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        # Shared encoder: progressively map in_channels -> mid_channels.
        # The first conv lifts channels; subsequent convs refine features
        # while keeping the channel width fixed.
        encoder_layers = []
        ch_in = in_channels
        for i in range(num_layers):
            ch_out = mid_channels
            encoder_layers.append(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1))
            encoder_layers.append(nn.ReLU(inplace=True))
            ch_in = ch_out
        self.encoder = nn.Sequential(*encoder_layers)

        # Reflectance head: predicts R in [0, 1].
        self.r_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, padding=1),
        )

        # Illumination head: predicts L in [0, 1].
        self.l_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, padding=1),
        )

        # Learnable per-channel affine correction for R that preserves the
        # [0, 1] sigmoid output.  Initialised to identity (gamma=1, beta=0).
        # We apply it BEFORE the final sigmoid in :meth:`forward`.
        self.r_gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.r_beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Decompose image ``x`` into ``(R, L)``.

        Args:
            x (Tensor): Input image ``(B, 3, H, W)`` in approximately
                ``[0, 1]``.

        Returns:
            Tuple[Tensor, Tensor]:

            - ``R`` (Tensor): Reflectance ``(B, 3, H, W)`` in ``[0, 1]``.
            - ``L`` (Tensor): Illumination ``(B, 3, H, W)`` in ``[0, 1]``.
        """
        feat = self.encoder(x)

        # Reflectance: affine correction before sigmoid keeps the [0, 1]
        # bound while still allowing the network to subtract residual
        # illumination bias.
        r_logits = self.r_head(feat)
        r_logits = self.r_gamma * r_logits + self.r_beta
        R = torch.sigmoid(r_logits)

        # Illumination: sigmoid bounds L to (0, 1).
        L = torch.sigmoid(self.l_head(feat))

        return R, L


class IlluminationTokenEncoder(BaseModule):
    """Encode the illumination map ``L`` into a compact token sequence.

    A tiny CNN that pools the illumination map into a fixed-size token
    sequence that can be prepended to the DETR encoder feature sequence.

    Args:
        in_channels (int): Channels of L (3 for RGB illumination).
            Defaults to 3.
        out_channels (int): Token embedding dimension (matches DETR hidden
            dim). Defaults to 256.
        num_tokens (int): Number of illumination tokens. Defaults to 4.
        init_cfg (dict, optional): Initialization config.
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
        """Encode the illumination map into tokens.

        Args:
            L (Tensor): Illumination map ``(B, 3, H, W)``.

        Returns:
            Tensor: Token sequence ``(B, num_tokens, out_channels)``.
        """
        feat = self.encoder(L)  # (B, C//2, 1, num_tokens)
        feat = feat.flatten(2).transpose(1, 2)  # (B, num_tokens, C//2)
        tokens = self.proj(feat)  # (B, num_tokens, out_channels)
        return tokens


@MODELS.register_module()
class RetinexDecomposedBackbone(BaseModule):
    """Retinex-Decomposed Backbone for Dark-DINO.

    The input image is first decomposed into R (reflectance) and L
    (illumination) by a lightweight :class:`RetinexDecomposer`.  R is then
    fed into a standard detection backbone (ResNet / Swin) whose multi-scale
    features are returned for detection.  L is encoded into illumination
    tokens for the DETR encoder.

    The decomposition head shares no parameters with the detection backbone,
    so pretrained backbone weights remain intact.

    Args:
        backbone (dict): Config of the detection backbone (e.g. ResNet-50).
        decomposer (dict, optional): Config of :class:`RetinexDecomposer`.
            Defaults to a 3-layer / 32-channel decomposer.
        light_encoder (dict, optional): Config of
            :class:`IlluminationTokenEncoder`.  Defaults to 4 tokens / 256-d.
        freeze_decomposer (bool): If True, freeze the decomposition head
            (used in stage-2/3 after stage-1 pretraining). Defaults to False.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(
        self,
        backbone: dict,
        decomposer: OptConfigType = None,
        light_encoder: OptConfigType = None,
        freeze_decomposer: bool = False,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        if decomposer is None:
            decomposer = dict()
        if light_encoder is None:
            light_encoder = dict()

        # Decomposition modules
        self.decomposer = RetinexDecomposer(**decomposer)
        self.light_encoder = IlluminationTokenEncoder(**light_encoder)

        # Detection backbone (built from mmdet registry so that pretrained
        # weights are automatically loaded via init_cfg of the inner module).
        self.backbone = MODELS.build(backbone)

        self.freeze_decomposer = freeze_decomposer
        if freeze_decomposer:
            self._freeze_decomposer()

    def _freeze_decomposer(self) -> None:
        """Freeze the decomposition head and illumination encoder."""
        for m in (self.decomposer, self.light_encoder):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def forward(
        self, x: Tensor
    ) -> Tuple[Tuple[Tensor, ...], Tensor, Tensor, Tensor]:
        """Forward.

        Args:
            x (Tensor): Input dark image ``(B, 3, H, W)``.

        Returns:
            Tuple containing:

            - ``mlvl_feats`` (tuple[Tensor]): Multi-scale features from the
              detection backbone applied on R.
            - ``light_tokens`` (Tensor): ``(B, num_tokens, C)`` illumination
              tokens.
            - ``R`` (Tensor): ``(B, 3, H, W)`` reflectance.
            - ``L`` (Tensor): ``(B, 3, H, W)`` illumination.
        """
        # 1. Retinex decomposition
        R, L = self.decomposer(x)

        # 2. Feed R into detection backbone (gradients flow back through R
        #    into the decomposer, jointly trained with detection loss).
        mlvl_feats = self.backbone(R)

        # 3. Encode L into tokens.  L is detached so that detection gradients
        #    do not corrupt the illumination estimate; the Retinex
        #    consistency loss is the only supervisor of L.
        light_tokens = self.light_encoder(L.detach())

        return mlvl_feats, light_tokens, R, L

    def train(self, mode: bool = True) -> 'RetinexDecomposedBackbone':
        """Override to keep frozen modules in eval mode.

        Args:
            mode (bool): Whether to set training mode (True) or evaluation
                mode (False). Defaults to True.
        """
        super().train(mode)
        # If the decomposer is frozen, force it into eval mode regardless
        # of the parent's training state (so BN/Dropout don't update stats).
        if self.freeze_decomposer:
            self.decomposer.eval()
            self.light_encoder.eval()
        return self
