# Copyright (c) OpenMMLab. All rights reserved.
"""DINOv2 Teacher Wrapper for Cross-Domain Distillation.

Loads a frozen DINOv2-ViT-L/14 model (via torch.hub or local checkpoint)
and provides an ``extract_features`` interface that returns multi-level
features suitable for the FeatureDistillationLoss.

The teacher is always in eval mode with ``requires_grad_(False)`` and is
NOT included in the optimizer.  It adds zero inference cost.

This is an ``nn.Module`` so that DDP / device placement are handled
automatically by mmengine.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class DINOv2Teacher(nn.Module):
    """DINOv2 teacher feature extractor (frozen, no grad).

    Args:
        model_name (str): DINOv2 model name for ``torch.hub.load``.
            Default ``'dinov2_vitl14'``.
        checkpoint_path (str, optional): Local path to a pretrained
            checkpoint. If provided, loaded via ``state_dict`` instead of
            ``torch.hub``. Default ``None``.
        output_layers (list[int], optional): Indices of transformer layers
            whose outputs are returned as multi-level features. Default
            ``[8, 16, 20, 24]`` for ViT-L (24 layers total).
    """

    def __init__(
        self,
        model_name: str = 'dinov2_vitl14',
        checkpoint_path: Optional[str] = None,
        output_layers: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.output_layers = output_layers or [8, 16, 20, 24]
        self._model_name = model_name
        self._checkpoint_path = checkpoint_path
        self._model: Optional[nn.Module] = None
        self._load_attempted = False
        # Hook handles — registered once, reused across forward calls.
        self._hook_handles: list = []
        self._hook_outputs: dict = {}

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Try to load the DINOv2 model (called once on first forward)."""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            if self._checkpoint_path is not None:
                model = torch.hub.load(
                    'facebookresearch/dinov2',
                    self._model_name,
                    pretrained=False)
                state_dict = torch.load(
                    self._checkpoint_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
            else:
                model = torch.hub.load(
                    'facebookresearch/dinov2',
                    self._model_name,
                    pretrained=True)
        except Exception as e:
            logger.warning(
                'Failed to load DINOv2 teacher: %s. '
                'Distillation will be disabled.', e)
            return

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self._model = model

        # Register hooks once.
        self._register_hooks()

    @property
    def model(self) -> Optional[nn.Module]:
        """Lazy property that loads the model on first access."""
        if self._model is None and not self._load_attempted:
            self._load_model()
        return self._model

    # ------------------------------------------------------------------
    # Persistent hooks (registered once, cleared each forward)
    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        """Register forward hooks on the specified transformer blocks."""
        if self._model is None:
            return

        def _hook_fn(layer_idx: int):
            def fn(module, input, output):
                self._hook_outputs[layer_idx] = output
            return fn

        for layer_idx in self.output_layers:
            block = self._model.blocks[layer_idx]
            handle = block.register_forward_hook(_hook_fn(layer_idx))
            self._hook_handles.append(handle)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_features(self, x: Tensor) -> List[Tensor]:
        """Extract multi-level features from the DINOv2 teacher.

        Args:
            x (Tensor): Normal-light images ``(B, 3, H, W)``, normalised
                with ImageNet mean/std.

        Returns:
            list[Tensor]: Features from the specified output layers, each
                ``(B, N, C)`` where ``N = (H/14)*(W/14)`` and ``C = 1024``
                for ViT-L.
        """
        if self.model is None:
            return []

        # Clear previous hook outputs.
        self._hook_outputs.clear()

        # Forward pass — hooks populate self._hook_outputs.
        _ = self.model(x)

        # Collect in order.
        features = []
        for layer_idx in self.output_layers:
            if layer_idx in self._hook_outputs:
                features.append(self._hook_outputs[layer_idx])

        return features

    def forward(self, x: Tensor) -> List[Tensor]:
        """Alias for :meth:`extract_features` (called by DarkDINO)."""
        return self.extract_features(x)

    # ------------------------------------------------------------------
    # Override train() to keep teacher frozen
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> 'DINOv2Teacher':
        super().train(mode)
        if self._model is not None:
            self._model.eval()
        return self
