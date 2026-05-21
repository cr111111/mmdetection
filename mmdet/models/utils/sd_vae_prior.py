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


class SDVAEPrior(nn.Module):
    """Stable Diffusion VAE encoder wrapper (frozen ``nn.Module``).

    Being an ``nn.Module`` ensures correct DDP device management and
    ``state_dict`` handling.

    Args:
        model_name (str): HuggingFace diffusers model name.
            Default ``'stabilityai/sd-vae-ft-mse'``.
        input_range (str): Value range of the *input* to
            :meth:`encode_features`.  ``'[-1,1]'`` means inputs are already
            in SD convention; ``'[0,1]'`` means they will be linearly
            mapped to ``[-1, 1]``.  Default ``'[0,1]'``.
    """

    def __init__(
        self,
        model_name: str = 'stabilityai/sd-vae-ft-mse',
        input_range: str = '[0,1]',
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.input_range = input_range
        self._vae: Optional[nn.Module] = None
        self._load_attempted = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _load_vae(self) -> None:
        """Try to load the SD VAE (called once on first forward)."""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(self.model_name)
            vae.eval()
            for p in vae.parameters():
                p.requires_grad = False
            self._vae = vae
            logger.info('Loaded SD VAE from %s', self.model_name)
        except ImportError:
            logger.warning(
                'diffusers package not found. Install with: '
                'pip install diffusers. DiffPrior will be disabled.')
        except Exception as e:
            logger.warning(
                'Failed to load SD VAE: %s. DiffPrior will be disabled.', e)

    @property
    def vae(self) -> Optional[nn.Module]:
        """Lazy property that loads the VAE on first access."""
        if self._vae is None and not self._load_attempted:
            self._load_vae()
        return self._vae

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_features(self, x: Tensor) -> Optional[Tensor]:
        """Extract VAE latent features from input images.

        Args:
            x (Tensor): Input images ``(B, 3, H, W)``.

        Returns:
            Tensor or None: Latent features ``(B, 4, H/8, W/8)``, or
            ``None`` if the VAE failed to load.
        """
        if self.vae is None:
            return None

        # Convert to [-1, 1] if input is in [0, 1].
        if self.input_range == '[0,1]':
            x = 2.0 * x - 1.0

        latent_dist = self.vae.encode(x)
        latent = latent_dist.latent_dist.sample()
        # Scale by 0.18215 (SD VAE scaling factor)
        latent = latent * 0.18215
        return latent

    # ------------------------------------------------------------------
    # Override train() to keep VAE frozen
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> 'SDVAEPrior':
        super().train(mode)
        if self._vae is not None:
            self._vae.eval()
        return self
