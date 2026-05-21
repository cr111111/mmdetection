# Copyright (c) OpenMMLab. All rights reserved.
"""Dark-DINO: low-light object detection via physics--frequency--generative
--semantic decomposition.

Dark-DINO extends DINO with four orthogonal innovations:

1. **Retinex-Decomposed Backbone (RD-Backbone)** - physics-aware ``I = R * L``
   decomposition; R drives detection, L drives illumination tokens.
2. **Frequency-Decoupled Neck (FreqDec-Neck)** - DCT 3-band decomposition
   with specialised experts and brightness-driven gating.
3. **Diffusion-Prior Feature Enhancement (DiffPrior-FPN)** - frozen Stable
   Diffusion VAE encoder provides illumination-invariant semantic priors.
4. **Cross-Domain DINOv2 Distillation (CD-Distill)** - aligns dark features
   with normal-light teacher features from paired data.

Innovations 3 and 4 are config-driven optional modules used for ablations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import OptConfigType
from .dino import DINO

logger = logging.getLogger(__name__)


@MODELS.register_module()
class DarkDINO(DINO):
    """Dark-DINO detector for low-light object detection.

    Inherits from :class:`DINO` and overrides ``extract_feat`` / ``loss`` to
    integrate the Retinex decomposition, frequency-decoupled neck, diffusion
    prior, and DINOv2 distillation.

    Args:
        retinex_loss (dict, optional): Config of
            :class:`RetinexConsistencyLoss`. Defaults to None (disabled).
        distill_loss (dict, optional): Config of
            :class:`FeatureDistillationLoss`. Defaults to None (disabled).
        diff_prior (dict, optional): Config of :class:`DiffPriorFPN`.
            Defaults to None (disabled).
        teacher_cfg (dict, optional): Config of :class:`DINOv2Teacher` used
            for distillation. Defaults to None.
        freq_gate_reg_weight (float): Regularisation weight for the
            frequency-gate weights (encourages non-degenerate distributions).
            Defaults to 0.0 (disabled).
    """

    def __init__(
        self,
        *args,
        retinex_loss: OptConfigType = None,
        distill_loss: OptConfigType = None,
        diff_prior: OptConfigType = None,
        teacher_cfg: OptConfigType = None,
        freq_gate_reg_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.freq_gate_reg_weight = float(freq_gate_reg_weight)

        # Retinex consistency loss.
        self.retinex_loss_fn = (
            MODELS.build(retinex_loss) if retinex_loss is not None else None)

        # Feature distillation loss.
        self.distill_loss_fn = (
            MODELS.build(distill_loss) if distill_loss is not None else None)

        # Diffusion prior module.
        self.diff_prior = (
            MODELS.build(diff_prior) if diff_prior is not None else None)

        # DINOv2 teacher (frozen).  Built eagerly so that DDP / device
        # placement work correctly; `requires_grad` is already disabled
        # inside the teacher wrapper.
        self.dinov2_teacher: Optional[nn.Module] = None
        if teacher_cfg is not None:
            self.dinov2_teacher = MODELS.build(teacher_cfg)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def extract_feat(
        self, batch_inputs: Tensor
    ) -> Tuple[Tuple[Tensor, ...], Optional[Tensor], Optional[Tensor],
               Optional[Tensor]]:
        """Extract features with Retinex decomposition + FreqDec neck.

        Args:
            batch_inputs (Tensor): Dark images ``(B, 3, H, W)``.

        Returns:
            Tuple containing:

            - ``mlvl_feats`` (tuple[Tensor]): multi-level features for the
              DINO encoder/decoder.
            - ``light_tokens`` (Tensor | None): illumination tokens.
            - ``R`` (Tensor | None): reflectance map.
            - ``L`` (Tensor | None): illumination map.
        """
        light_tokens: Optional[Tensor] = None
        R: Optional[Tensor] = None
        L: Optional[Tensor] = None

        # Step 1: backbone (with optional Retinex decomposition).
        if hasattr(self.backbone, 'decomposer'):
            mlvl_feats_raw, light_tokens, R, L = self.backbone(batch_inputs)
        else:
            mlvl_feats_raw = self.backbone(batch_inputs)

        # Step 2: neck (FreqDec accepts illumination map; vanilla necks
        # don't).  We dispatch by checking the neck's forward signature.
        if self.with_neck:
            if self._neck_accepts_light():
                mlvl_feats = self.neck(
                    mlvl_feats_raw,
                    light_tokens=light_tokens,
                    light_map=L)
            else:
                mlvl_feats = self.neck(mlvl_feats_raw)
        else:
            mlvl_feats = mlvl_feats_raw

        # Step 3: optional diffusion-prior fusion.
        if self.diff_prior is not None:
            mlvl_feats = self.diff_prior(mlvl_feats, batch_inputs)

        # Cache backbone-level features for distillation (avoids a second
        # forward pass).  Only stored during training.
        if self.training:
            self._student_feats_cache: Tuple[Tensor, ...] = mlvl_feats_raw
        else:
            self._student_feats_cache = None

        return mlvl_feats, light_tokens, R, L

    def _neck_accepts_light(self) -> bool:
        """Whether ``self.neck.forward`` accepts ``light_map`` kwarg."""
        try:
            varnames = self.neck.forward.__code__.co_varnames
        except AttributeError:
            return False
        return 'light_map' in varnames

    # ------------------------------------------------------------------
    # Training: loss
    # ------------------------------------------------------------------
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Compute DINO detection losses + Dark-DINO auxiliary losses.

        Args:
            batch_inputs (Tensor): Dark images ``(B, 3, H, W)``.
            batch_data_samples (list): Data samples with GT annotations and
                (optionally) paired normal-light images.

        Returns:
            Dict[str, Tensor]: Dictionary of all loss components.
        """
        mlvl_feats, light_tokens, R, L = self.extract_feat(batch_inputs)

        # Standard DINO transformer forward + detection loss.
        head_inputs_dict = self.forward_transformer(
            mlvl_feats, batch_data_samples)
        losses: Dict[str, Tensor] = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        # ----------------------- auxiliary losses ----------------------
        # Retinex consistency.
        if (self.retinex_loss_fn is not None
                and R is not None and L is not None):
            I_raw = self._unnormalize(batch_inputs)
            losses.update(self.retinex_loss_fn(I_raw, R, L))

        # Cross-domain DINOv2 distillation.
        if (self.distill_loss_fn is not None
                and self.dinov2_teacher is not None):
            normal_imgs = self._get_paired_normal_images(
                batch_data_samples, device=batch_inputs.device)
            student_feats = self._get_student_features()
            if normal_imgs is not None and student_feats:
                with torch.no_grad():
                    teacher_feats = self.dinov2_teacher(normal_imgs)
                losses.update(
                    self.distill_loss_fn(student_feats, teacher_feats))

        # Frequency-gate regularisation: penalise large gate magnitudes so
        # that the softmax over bands stays non-degenerate.  We accumulate
        # over all gate parameters into a single scalar (fixing the bug
        # where the previous version overwrote the entry inside a loop).
        if self.freq_gate_reg_weight > 0 and self._has_brightness_gate():
            reg = batch_inputs.new_zeros(())
            n = 0
            for name, param in self.neck.brightness_gate.named_parameters():
                if 'weight' in name:
                    reg = reg + param.abs().mean()
                    n += 1
            if n > 0:
                losses['loss_freq_gate_reg'] = (
                    self.freq_gate_reg_weight * reg / n)

        return losses

    def _has_brightness_gate(self) -> bool:
        return (self.with_neck
                and hasattr(self.neck, 'brightness_gate')
                and isinstance(self.neck.brightness_gate, nn.Module))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict without auxiliary losses (inference mode).

        The Retinex decomposition, FreqDec neck and DiffPrior are still
        applied (they are part of the network).  No teacher / SD models
        are used at inference time.
        """
        mlvl_feats, _, _, _ = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            mlvl_feats, batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward without post-processing (for tracing / debug)."""
        mlvl_feats, _, _, _ = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            mlvl_feats, batch_data_samples)
        return self.bbox_head.forward(**head_inputs_dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _unnormalize(self, x: Tensor) -> Tensor:
        """Undo :class:`DetDataPreprocessor` normalisation, returning images
        in ``[0, 1]``.

        Args:
            x (Tensor): Normalised images ``(B, 3, H, W)``.

        Returns:
            Tensor: Images in ``[0, 1]``.
        """
        dp = getattr(self, 'data_preprocessor', None)
        if dp is not None and hasattr(dp, 'mean') and dp.mean is not None:
            mean = dp.mean.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            std = dp.std.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            x = x * std + mean
        # Pixel range was [0, 255] before normalisation.
        return (x / 255.0).clamp(0.0, 1.0)

    def _get_paired_normal_images(
        self,
        batch_data_samples: SampleList,
        device: torch.device,
    ) -> Optional[Tensor]:
        """Return paired normal-light images, or ``None`` if unavailable.

        Paired images are expected to be attached to each data sample as
        ``sample.normal_img`` (a CHW tensor) by a paired data-preprocessor
        / dataset.

        Args:
            batch_data_samples (SampleList): Per-sample annotations.
            device (torch.device): Target device for the stacked tensor.
        """
        if not batch_data_samples:
            return None
        if not all(hasattr(s, 'normal_img') for s in batch_data_samples):
            return None
        imgs = torch.stack([s.normal_img for s in batch_data_samples], dim=0)
        return imgs.to(device=device, dtype=next(self.parameters()).dtype)

    def _get_student_features(self) -> List[Tensor]:
        """Return the cached student multi-level features for distillation.

        Populated by :meth:`extract_feat` during training.  Returns an empty
        list if the cache is missing (e.g. during the first forward).
        """
        feats = getattr(self, '_student_feats_cache', None)
        if feats is None:
            return []
        return list(feats)
