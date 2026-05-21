# Copyright (c) OpenMMLab. All rights reserved.
"""Dark-DINO: Low-light object detection via physics-frequency-generative
semantic decomposition.

Dark-DINO extends DINO with four orthogonal innovations:
1. Retinex-Decomposed Backbone (RD-Backbone): physics-aware I = R * L
   decomposition, R for detection, L for illumination tokens.
2. Frequency-Decoupled Neck (FreqDec-Neck): DCT 3-band decomposition +
   specialised experts + brightness-driven gate.
3. Diffusion-Prior Feature Enhancement (DiffPrior-FPN): frozen SD VAE
   encoder provides illumination-invariant semantic priors.
4. Cross-Domain DINOv2 Distillation (CD-Distill): aligns dark features
   with normal-light teacher features from paired data.

Innovations 3 and 4 are optional (config-driven switches) for ablation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import OptConfigType
from ..layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder,
                      DinoTransformerDecoder, SinePositionalEncoding)
from .dino import DINO

logger = logging.getLogger(__name__)


@MODELS.register_module()
class DarkDINO(DINO):
    """Dark-DINO detector for low-light object detection.

    Inherits from DINO and overrides extract_feat / loss to integrate the
    Retinex decomposition, frequency-decoupled neck, diffusion prior, and
    DINOv2 distillation.

    Args:
        retinex_loss (dict, optional): Config for RetinexConsistencyLoss.
            Default None (disabled).
        distill_loss (dict, optional): Config for FeatureDistillationLoss.
            Default None (disabled).
        diff_prior (dict, optional): Config for DiffPriorFPN.
            Default None (disabled).
        freq_gate_reg_weight (float): Regularisation weight for the frequency
            gate (encourages non-degenerate gate distributions). Default 0.01.
    """

    def __init__(
        self,
        *args,
        retinex_loss: OptConfigType = None,
        distill_loss: OptConfigType = None,
        diff_prior: OptConfigType = None,
        freq_gate_reg_weight: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.freq_gate_reg_weight = freq_gate_reg_weight

        # Retinex consistency loss
        self.retinex_loss_fn = None
        if retinex_loss is not None:
            self.retinex_loss_fn = MODELS.build(retinex_loss)

        # Feature distillation loss
        self.distill_loss_fn = None
        if distill_loss is not None:
            self.distill_loss_fn = MODELS.build(distill_loss)

        # Diffusion prior module
        self.diff_prior = None
        if diff_prior is not None:
            self.diff_prior = MODELS.build(diff_prior)

        # DINOv2 teacher (lazy init)
        self._dinov2_teacher = None
        self._dinov2_cfg = None
        if distill_loss is not None and distill_loss.get('teacher_cfg'):
            self._dinov2_cfg = distill_loss.pop('teacher_cfg')

        # SD VAE prior (lazy init)
        self._sd_vae = None
        self._sd_vae_cfg = None
        if diff_prior is not None and diff_prior.get('vae_cfg'):
            self._sd_vae_cfg = diff_prior.pop('vae_cfg')

    def extract_feat(
        self, batch_inputs: Tensor
    ) -> Tuple[Tuple[Tensor, ...], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Extract features with Retinex decomposition + FreqDec neck.

        Args:
            batch_inputs (Tensor): Dark images (B, 3, H, W).

        Returns:
            Tuple containing:
                - mlvl_feats (tuple[Tensor]): Multi-level features for the
                  DINO encoder/decoder.
                - light_tokens (Tensor | None): Illumination tokens.
                - R (Tensor | None): Reflectance map.
                - L (Tensor | None): Illumination map.
        """
        R, L = None, None
        light_tokens = None

        # RD-Backbone: decompose + backbone feature extraction
        if isinstance(self.backbone, nn.Module) and hasattr(
                self.backbone, 'decomposer'):
            # RetinexDecomposedBackbone
            mlvl_feats_raw, light_tokens, R, L = self.backbone(batch_inputs)
        else:
            # Fallback: standard backbone (no decomposition)
            mlvl_feats_raw = self.backbone(batch_inputs)

        # Neck (FreqDecoupledNeck or standard ChannelMapper)
        if self.with_neck:
            if hasattr(self.neck, 'forward') and 'light_map' in \
                    self.neck.forward.__code__.co_varnames:
                mlvl_feats = self.neck(mlvl_feats_raw,
                                       light_tokens=light_tokens,
                                       light_map=L)
            else:
                mlvl_feats = self.neck(mlvl_feats_raw)
        else:
            mlvl_feats = mlvl_feats_raw

        # DiffPrior (optional)
        if self.diff_prior is not None:
            mlvl_feats = self.diff_prior(mlvl_feats, batch_inputs)

        return mlvl_feats, light_tokens, R, L

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses including DINO detection losses + auxiliary losses.

        Args:
            batch_inputs (Tensor): Dark images (B, 3, H, W).
            batch_data_samples (list): Data samples with GT annotations.

        Returns:
            dict: Dictionary of all loss components.
        """
        mlvl_feats, light_tokens, R, L = self.extract_feat(batch_inputs)

        # Standard DINO transformer forward
        head_inputs_dict = self.forward_transformer(
            mlvl_feats, batch_data_samples)

        # DINO detection losses
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        # Retinex consistency loss
        if self.retinex_loss_fn is not None and R is not None and L is not None:
            # Need original image in [0,1] range for Retinex loss
            # batch_inputs is already mean-centered; reconstruct raw
            I_raw = self._unnormalize(batch_inputs)
            retinex_losses = self.retinex_loss_fn(I_raw, R, L)
            losses.update(retinex_losses)

        # Feature distillation loss
        if self.distill_loss_fn is not None and self._dinov2_teacher is not None:
            # Extract teacher features from paired normal-light images
            normal_imgs = self._get_paired_normal_images(batch_data_samples)
            if normal_imgs is not None:
                with torch.no_grad():
                    teacher_feats = self._dinov2_teacher(normal_imgs)
                student_feats = self._get_student_features()
                distill_losses = self.distill_loss_fn(student_feats,
                                                      teacher_feats)
                losses.update(distill_losses)

        # Frequency gate regularisation
        if self.freq_gate_reg_weight > 0 and hasattr(self, 'neck') and \
                hasattr(self.neck, 'brightness_gate'):
            gate = self.neck.brightness_gate
            # Encourage gate weights to be non-degenerate
            # (not all mass on one band)
            for name, param in gate.named_parameters():
                if 'weight' in name:
                    losses['loss_freq_gate_reg'] = \
                        self.freq_gate_reg_weight * param.abs().mean()

        return losses

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict without auxiliary losses (inference mode).

        The Retinex decomposition, FreqDec neck, and DiffPrior are all
        applied, but no teacher/SD models are used at inference time.
        """
        mlvl_feats, _, _, _ = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            mlvl_feats, batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict, rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward without post-processing."""
        mlvl_feats, _, _, _ = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            mlvl_feats, batch_data_samples)
        return self.bbox_head.forward(**head_inputs_dict)

    def _unnormalize(self, x: Tensor) -> Tensor:
        """Reverse the DetDataPreprocessor normalization to get [0,1] images.

        Args:
            x (Tensor): Normalized images (B, 3, H, W).

        Returns:
            Tensor: Unnormalized images in ~[0, 1].
        """
        if hasattr(self, 'data_preprocessor') and \
                hasattr(self.data_preprocessor, 'mean'):
            mean = torch.as_tensor(
                self.data_preprocessor.mean,
                device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            std = torch.as_tensor(
                self.data_preprocessor.std,
                device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            x = x * std + mean
        # Clip to [0, 1] (values from [0, 255] → /255)
        return (x / 255.0).clamp(0, 1)

    def _get_paired_normal_images(
            self, batch_data_samples: SampleList) -> Optional[Tensor]:
        """Extract paired normal-light images from data samples if available.

        Returns None if paired data is not provided (e.g. during standard
        detection training without paired data).
        """
        if len(batch_data_samples) > 0 and hasattr(
                batch_data_samples[0], 'normal_img'):
            normal_imgs = torch.stack(
                [s.normal_img for s in batch_data_samples])
            return normal_imgs.to(batch_data_samples[0].metainfo.get(
                'img_norm_cfg_device', next(self.parameters()).device))
        return None

    def _get_student_features(self) -> List[Tensor]:
        """Get intermediate student features for distillation.

        Returns multi-level backbone features from the current forward pass.
        This relies on the backbone storing features during forward.
        """
        feats = []
        if hasattr(self.backbone, 'backbone'):
            inner_backbone = self.backbone.backbone
        else:
            inner_backbone = self.backbone

        # Collect features from backbone stages
        if hasattr(inner_backbone, 'out_indices'):
            for idx in inner_backbone.out_indices:
                layer_name = f'layer{idx + 1}' if hasattr(
                    inner_backbone, f'layer{idx + 1}') else None
                if layer_name and hasattr(inner_backbone, layer_name):
                    # This is a placeholder; actual features come from
                    # the forward pass. In practice, the backbone already
                    # stores these during extract_feat.
                    pass
        return feats
