# Copyright (c) OpenMMLab. All rights reserved.
"""Feature Distillation Loss for Cross-Domain DINOv2 Distillation (CD-Distill).

This loss aligns features extracted by the student detector (on dark images)
with features from a frozen DINOv2 teacher (on paired normal-light images).

Two distillation objectives are supported:
1. **Cosine feature alignment**: minimises 1 - cos(student_feat, teacher_feat)
   for each spatial position.
2. **Relation-based KD**: aligns pair-wise relation matrices (soft cosine
   similarity matrices) between student and teacher features via KL divergence.

Both losses operate on multi-level features and can be individually weighted.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class FeatureDistillationLoss(BaseModule):
    """Cross-domain feature distillation loss.

    Args:
        cosine_weight (float): Weight for cosine alignment loss. Default 1.0.
        relation_weight (float): Weight for relation-based KD loss. Default 0.5.
        temperature (float): Temperature for softmax in relation KD.
            Default 4.0.
        num_levels (int): Number of feature levels to supervise. Default 4.
        student_channels (int): Channel dimension of student features.
            Default 256.
        teacher_channels (int): Channel dimension of teacher features.
            Default 1024 (DINOv2-ViT-L).
    """

    def __init__(
        self,
        cosine_weight: float = 1.0,
        relation_weight: float = 0.5,
        temperature: float = 4.0,
        num_levels: int = 4,
        student_channels: int = 256,
        teacher_channels: int = 1024,
    ) -> None:
        super().__init__()
        self.cosine_weight = cosine_weight
        self.relation_weight = relation_weight
        self.temperature = temperature
        self.num_levels = num_levels

        # Projection layers to align student → teacher channel dim
        self.proj_layers = nn.ModuleList([
            nn.Linear(student_channels, teacher_channels, bias=False)
            for _ in range(num_levels)
        ])

    def forward(
        self,
        student_feats: list,
        teacher_feats: list,
    ) -> dict:
        """Compute distillation losses.

        Args:
            student_feats (list[Tensor]): Student features at multiple levels,
                each (B, C_s, H, W) or (B, N, C_s).
            teacher_feats (list[Tensor]): Teacher features at corresponding
                levels, each (B, C_t, H', W') or (B, N', C_t).

        Returns:
            dict: Dictionary of loss tensors.
        """
        if not student_feats or not teacher_feats:
            return {}

        loss_cosine = torch.tensor(0.0, device=student_feats[0].device)
        loss_relation = torch.tensor(0.0, device=student_feats[0].device)

        num_pairs = min(len(student_feats), len(teacher_feats), self.num_levels)

        for i in range(num_pairs):
            s_feat = student_feats[i]  # (B, C_s, H, W)
            t_feat = teacher_feats[i]  # (B, C_t, H', W')

            # Handle different spatial sizes: interpolate teacher to student
            if s_feat.dim() == 4 and t_feat.dim() == 4:
                B, Cs, Hs, Ws = s_feat.shape
                Bt, Ct, Ht, Wt = t_feat.shape
                if Hs != Ht or Ws != Wt:
                    t_feat = F.interpolate(
                        t_feat, size=(Hs, Ws), mode='bilinear',
                        align_corners=False)

                # Reshape for projection: (B, H*W, C)
                s_flat = s_feat.flatten(2).transpose(1, 2)  # (B, N, Cs)
                t_flat = t_feat.flatten(2).transpose(1, 2)  # (B, N, Ct)
            else:
                s_flat = s_feat
                t_flat = t_feat

            # Project student features to teacher channel space
            s_proj = self.proj_layers[i](s_flat)  # (B, N, Ct)

            # 1. Cosine alignment loss
            cos_loss = 1.0 - F.cosine_similarity(
                s_proj, t_flat, dim=-1).mean()
            loss_cosine = loss_cosine + cos_loss

            # 2. Relation-based KD loss
            if self.relation_weight > 0:
                # Compute relation matrices (pair-wise cosine similarity)
                s_rel = self._relation_matrix(s_proj)  # (B, N, N)
                t_rel = self._relation_matrix(t_flat)  # (B, N, N)

                # KL divergence on softmax-ed relation matrices
                s_rel_log = F.log_softmax(
                    s_rel / self.temperature, dim=-1)
                t_rel_soft = F.softmax(
                    t_rel / self.temperature, dim=-1)
                rel_loss = F.kl_div(
                    s_rel_log, t_rel_soft, reduction='batchmean')
                loss_relation = loss_relation + rel_loss

        num_pairs = max(num_pairs, 1)
        # Return per-term weighted losses; mmengine will sum them.
        # We do NOT also return the aggregate to avoid double counting.
        return {
            'loss_distill_cosine': self.cosine_weight * loss_cosine / num_pairs,
            'loss_distill_relation': self.relation_weight * loss_relation / num_pairs,
        }

    @staticmethod
    def _relation_matrix(x: Tensor) -> Tensor:
        """Compute pair-wise cosine similarity matrix.

        Args:
            x (Tensor): (B, N, C) feature tokens.

        Returns:
            Tensor: (B, N, N) cosine similarity matrix.
        """
        x_norm = F.normalize(x, dim=-1)
        return torch.bmm(x_norm, x_norm.transpose(1, 2))
