# Copyright (c) OpenMMLab. All rights reserved.
"""Stable Diffusion VAE Encoder Wrapper for Diffusion-Prior Feature Enhancement.

Loads a frozen Stable Diffusion VAE encoder and extracts illumination-
invariant semantic priors from input images.  The VAE latent features are
then used as key/value in a cross-attention module that lets dark-image
features "query" the normal-light semantic world.

The SD VAE is always frozen (eval + no_grad) and adds ~83M parameters but
zero training cost for the VAE itself.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class SDVAEPrior:
    """Stable Diffusion VAE encoder wrapper (frozen).

    Args:
        model_name (str): HuggingFace diffusers model name.
            Default 'stabilityai/sd-vae-ft-mse'.
        device (str): Device to place the model on. Default 'cuda'.
        dtype (str): Data type for the model. Default 'float16'.
    """

    def __init__(
        self,
        model_name: str = 'stabilityai/sd-vae-ft-mse',
        device: str = 'cuda',
        dtype: str = 'float16',
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._vae = None

    def _load_vae(self) -> Optional[nn.Module]:
        """Lazy-load the SD VAE encoder."""
        try:
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(self.model_name)
            vae = vae.to(device=self.device, dtype=getattr(
                torch, self.dtype))
            vae.eval()
            for p in vae.parameters():
                p.requires_grad = False
            logger.info(f'Loaded SD VAE from {self.model_name}')
            return vae
        except ImportError:
            logger.warning(
                'diffusers package not found. Install with: '
                'pip install diffusers. DiffPrior will be disabled.')
            return None
        except Exception as e:
            logger.warning(
                f'Failed to load SD VAE: {e}. '
                f'DiffPrior will be disabled.')
            return None

    @property
    def vae(self) -> Optional[nn.Module]:
        """Lazy property that loads the VAE on first access."""
        if self._vae is None:
            self._vae = self._load_vae()
        return self._vae

    @torch.no_grad()
    def encode_features(self, x: Tensor) -> Optional[Tensor]:
        """Extract VAE latent features from input images.

        Args:
            x (Tensor): Input images (B, 3, H, W), in [-1, 1] range
                (SD VAE convention).

        Returns:
            Tensor or None: Latent features (B, 4, H/8, W/8), or None if
                the VAE failed to load.
        """
        if self.vae is None:
            return None

        # SD VAE expects input in [-1, 1]
        # If input is in [0, 1], transform: x = 2 * x - 1
        if x.min() >= 0 and x.max() <= 1:
            x = 2 * x - 1

        latent_dist = self.vae.encode(x)
        latent = latent_dist.latent_dist.sample()
        # Scale by 0.18215 (SD VAE scaling factor)
        latent = latent * 0.18215
        return latent

    def to(self, device: str) -> 'SDVAEPrior':
        """Move the VAE to a specific device."""
        self.device = device
        if self._vae is not None:
            self._vae = self._vae.to(device)
        return self
