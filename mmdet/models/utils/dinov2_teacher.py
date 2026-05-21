# Copyright (c) OpenMMLab. All rights reserved.
"""DINOv2 Teacher Wrapper for Cross-Domain Distillation.

Loads a frozen DINOv2-ViT-L/14 model (via torch.hub or local checkpoint)
and provides a ``extract_features`` interface that returns multi-level
features suitable for the FeatureDistillationLoss.

The teacher is always in eval mode with ``requires_grad_(False)`` and is
NOT included in the optimizer.  It adds zero inference cost.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class DINOv2Teacher:
    """DINOv2 teacher feature extractor (frozen, no grad).

    Args:
        model_name (str): DINOv2 model name for torch.hub.load.
            Default 'dinov2_vitl14'.
        checkpoint_path (str, optional): Local path to a pretrained
            checkpoint. If provided, loaded via state_dict instead of
            torch.hub. Default None.
        device (str): Device to place the model on. Default 'cuda'.
        output_layers (list[int]): Indices of transformer layers whose
            outputs are returned as multi-level features. Default
            [8, 16, 20, 24] for ViT-L (24 layers total).
    """

    def __init__(
        self,
        model_name: str = 'dinov2_vitl14',
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda',
        output_layers: Optional[List[int]] = None,
    ) -> None:
        self.device = device
        self.output_layers = output_layers or [8, 16, 20, 24]
        self._model = None
        self._model_name = model_name
        self._checkpoint_path = checkpoint_path

    def _load_model(self) -> nn.Module:
        """Lazy-load the DINOv2 model."""
        try:
            if self._checkpoint_path is not None:
                # Load from local checkpoint
                model = torch.hub.load(
                    'facebookresearch/dinov2',
                    self._model_name,
                    pretrained=False)
                state_dict = torch.load(
                    self._checkpoint_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
            else:
                # Load from torch hub (downloads on first use)
                model = torch.hub.load(
                    'facebookresearch/dinov2',
                    self._model_name,
                    pretrained=True)
        except Exception as e:
            logger.warning(
                f'Failed to load DINOv2 teacher: {e}. '
                f'Distillation will be disabled.')
            return None

        model = model.to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    @property
    def model(self) -> Optional[nn.Module]:
        """Lazy property that loads the model on first access."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @torch.no_grad()
    def extract_features(self, x: Tensor) -> List[Tensor]:
        """Extract multi-level features from the DINOv2 teacher.

        Args:
            x (Tensor): Normal-light images (B, 3, H, W), normalised with
                ImageNet mean/std.

        Returns:
            list[Tensor]: Features from the specified output layers, each
                (B, N, C) where N = (H/14)*(W/14) and C = 1024 for ViT-L.
        """
        if self.model is None:
            return []

        B = x.size(0)
        features = []

        # DINOv2 ViT forward with intermediate features
        # We hook into the transformer blocks to extract intermediate outputs
        hooks = []
        hook_outputs = {}

        def hook_fn(layer_idx):
            def fn(module, input, output):
                hook_outputs[layer_idx] = output
            return fn

        for layer_idx in self.output_layers:
            block = self.model.blocks[layer_idx]
            handle = block.register_forward_hook(hook_fn(layer_idx))
            hooks.append(handle)

        # Forward pass
        _ = self.model(x)

        # Collect hooked outputs
        for layer_idx in self.output_layers:
            if layer_idx in hook_outputs:
                feat = hook_outputs[layer_idx]  # (B, N, C)
                features.append(feat)

        # Remove hooks
        for h in hooks:
            h.remove()

        return features

    def to(self, device: str) -> 'DINOv2Teacher':
        """Move the teacher model to a specific device."""
        self.device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self
