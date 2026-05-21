# Copyright (c) OpenMMLab. All rights reserved.
"""Low-contrast adaptive Deformable DETR.

This detector is a thin subclass of :class:`DeformableDETR` that only
overrides :meth:`pre_decoder`. The motivation: in the vanilla single-stage
Deformable DETR, all reference points are produced from a learnable
``query_embedding`` via a linear projection, *regardless of the input image
content*. This is sub-optimal in the low-contrast detection setting where
the targets concentrate in regions of small intra-feature variance.

We compute, from the encoder ``memory``, a per-position **low-contrast
score** (smaller channel-variance ⇒ higher score) and use the *Top-K
low-variance positions* as data-dependent priors for a fraction of the
reference points. The remaining reference points still come from the
original learnable path, so the model retains the diversity of vanilla
Deformable DETR.

Mathematically, for each batch we:

1. Reshape ``memory`` of shape ``(B, N, C)`` into per-level 2-D maps using
   ``spatial_shapes`` and pick a single level (the highest-resolution one,
   since low-contrast targets benefit most from it).
2. Compute the per-position channel std and pick the ``K_low`` positions
   with the *smallest* std.
3. Map their grid indices to normalised coordinates ``(cx, cy) ∈ (0, 1)``,
   which are valid initial reference points for a single-stage Deformable
   DETR decoder (whose reference points are 2-D).
4. Replace the first ``K_low`` rows of ``reference_points`` with these
   coordinates; the rest are kept as in the vanilla detector.

The override is a no-op when ``as_two_stage=True`` — the parent
implementation is used directly, since two-stage already adapts to the input
image via the encoder proposal module.
"""
from typing import Dict, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from .deformable_detr import DeformableDETR


@MODELS.register_module()
class LowContrastDeformableDETR(DeformableDETR):
    """Deformable DETR with content-aware low-contrast query initialisation.

    Args:
        low_contrast_ratio (float): Fraction of queries whose initial
            reference points are placed at low-variance positions of the
            encoder memory. Must be in ``[0, 1]``. ``0`` reproduces the
            vanilla single-stage Deformable DETR. Default 0.3.
        low_contrast_level (int): Index of the encoder feature level used to
            score positions. Default 0 (highest-resolution level), which is
            the most informative for small / low-contrast objects.
    """

    def __init__(self,
                 *args,
                 low_contrast_ratio: float = 0.3,
                 low_contrast_level: int = 0,
                 **kwargs) -> None:
        assert 0.0 <= low_contrast_ratio <= 1.0, (
            f'low_contrast_ratio must be in [0, 1], got {low_contrast_ratio}.')
        self.low_contrast_ratio = low_contrast_ratio
        self.low_contrast_level = low_contrast_level
        super().__init__(*args, **kwargs)

    def _low_variance_reference_points(
            self,
            memory: Tensor,
            memory_mask: Tensor,
            spatial_shapes: Tensor,
            num_low: int) -> Tensor:
        """Pick ``num_low`` low-variance positions per batch as 2-D refs.

        Args:
            memory (Tensor): Encoder output ``(B, N, C)``.
            memory_mask (Tensor): Padding mask ``(B, N)`` (True = padded) or
                None.
            spatial_shapes (Tensor): ``(num_levels, 2)`` (h, w) per level.
            num_low (int): Number of reference points to pick.

        Returns:
            Tensor: ``(B, num_low, 2)`` normalised ``(cx, cy)`` coordinates
            in ``(0, 1)``.
        """
        bs = memory.size(0)
        device = memory.device

        # Locate the slice of memory belonging to ``low_contrast_level``.
        level_idx = max(0, min(self.low_contrast_level,
                               spatial_shapes.size(0) - 1))
        h_lvl = int(spatial_shapes[level_idx, 0].item())
        w_lvl = int(spatial_shapes[level_idx, 1].item())
        # Compute the start offset by summing previous levels' h*w.
        if level_idx == 0:
            start = 0
        else:
            start = int(spatial_shapes[:level_idx, 0]
                        .mul(spatial_shapes[:level_idx, 1]).sum().item())
        end = start + h_lvl * w_lvl

        # (B, h*w, C)
        mem_lvl = memory[:, start:end, :]
        # Per-position channel std, smaller = lower contrast.
        std = mem_lvl.std(dim=-1)  # (B, h*w)

        if memory_mask is not None:
            mask_lvl = memory_mask[:, start:end]
            # Padded positions get +inf so they are never selected.
            std = std.masked_fill(mask_lvl, float('inf'))

        # Top-K *smallest* std values per batch.
        k = min(num_low, std.size(1))
        _, idx = torch.topk(std, k=k, dim=1, largest=False)  # (B, k)

        # Convert flat idx back to (y, x) grid coordinates.
        y_idx = (idx // w_lvl).float()
        x_idx = (idx % w_lvl).float()
        # Centre-of-cell normalisation, same convention as
        # ``gen_encoder_output_proposals``.
        cx = (x_idx + 0.5) / max(w_lvl, 1)
        cy = (y_idx + 0.5) / max(h_lvl, 1)
        ref = torch.stack([cx, cy], dim=-1)  # (B, k, 2)

        # Pad with the centre point if k < num_low (extreme small features).
        if k < num_low:
            pad = ref.new_full((bs, num_low - k, 2), 0.5)
            ref = torch.cat([ref, pad], dim=1)

        # Clamp to a safe open interval so that ``inverse_sigmoid`` is finite
        # and gradient flow stays healthy.
        ref = ref.clamp(min=1e-3, max=1 - 1e-3)
        return ref.to(device)

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor) -> Tuple[Dict, Dict]:
        """Override: inject low-variance content-aware reference points.

        Falls back to the parent implementation when ``as_two_stage=True`` or
        when ``low_contrast_ratio=0``.
        """
        # Two-stage already adapts to the input image; nothing to do.
        if self.as_two_stage or self.low_contrast_ratio <= 0.0:
            return super().pre_decoder(memory, memory_mask, spatial_shapes)

        batch_size, _, c = memory.shape
        # ----- vanilla single-stage path (mirrors DeformableDETR.pre_decoder)
        enc_outputs_class, enc_outputs_coord = None, None
        query_embed = self.query_embedding.weight
        query_pos, query = torch.split(query_embed, c, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
        query = query.unsqueeze(0).expand(batch_size, -1, -1)
        # Original reference points from learnable embeddings.
        reference_points = self.reference_points_fc(query_pos).sigmoid()

        # ----- low-contrast adaptive injection
        num_low = int(round(self.num_queries * self.low_contrast_ratio))
        if num_low > 0:
            low_refs = self._low_variance_reference_points(
                memory=memory,
                memory_mask=memory_mask,
                spatial_shapes=spatial_shapes,
                num_low=num_low)
            # Detach the data-driven reference points so the gradient flows
            # only through the decoder/head, not back through the encoder
            # std statistics — this avoids a noisy second-order signal.
            low_refs = low_refs.detach()
            # Replace the first ``num_low`` reference points.
            reference_points = reference_points.clone()
            reference_points[:, :num_low, :] = low_refs

        decoder_inputs_dict = dict(
            query=query,
            query_pos=query_pos,
            memory=memory,
            reference_points=reference_points)
        head_inputs_dict = dict(
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict
