# Copyright (c) OpenMMLab. All rights reserved.
"""Contrast-Aware Focal Loss for low-contrast object detection.

Standard Focal Loss down-weights *easy* samples by ``(1 - p_t)**gamma``. In
low-contrast scenes, however, foreground and background distributions overlap
heavily, so the predicted probability ``p_t`` of a *true positive* is often
small and the ``(1 - p_t)**gamma`` factor stays close to 1 — but for
*hard background* points it is also close to 1, so the loss is dominated by
the abundant background.

This loss introduces a per-position **contrast modulation factor** that
adaptively reduces the effective ``gamma`` at low-contrast regions, making
the loss surface flatter there and giving low-contrast positives a larger
gradient.

The contrast signal is derived directly from the predicted logits' per-row
variance over the class dimension, which acts as a *zero-cost* online proxy
for confidence dispersion: a position whose logits are flat across classes
is exactly the kind of low-contrast / ambiguous position we want to up-weight.

The loss is fully backward-compatible: when ``contrast_gamma_scale=0`` it
collapses to the standard ``FocalLoss``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from .utils import weight_reduce_loss


def _compute_contrast_weight(pred: torch.Tensor) -> torch.Tensor:
    """Estimate a per-position low-contrast weight in [0, 1].

    The weight is high (→1) where the predicted logits are flat across the
    class dimension (low cls dispersion = ambiguous = low-contrast-like) and
    low (→0) where logits are spiky (confident).

    Args:
        pred (Tensor): Logits of shape ``(N, C)`` where ``N`` is the total
            number of predictions and ``C`` the number of classes.

    Returns:
        Tensor: ``(N, 1)`` low-contrast weight, *detached* from the graph
        (used purely as a modulation signal).
    """
    with torch.no_grad():
        # Use sigmoid because the head is a sigmoid focal classifier.
        prob = pred.sigmoid()
        # Spread of probabilities across classes. Low std → flat → ambiguous.
        spread = prob.std(dim=1, keepdim=True)  # (N, 1)
        # Normalise per-batch so the absolute scale of `spread` is not
        # detector-specific. Use a small eps for numerical stability.
        spread_max = spread.max().clamp_min(1e-6)
        spread_norm = spread / spread_max          # in [0, 1]
        contrast_w = (1.0 - spread_norm).clamp(0.0, 1.0)  # high = low-contrast
    return contrast_w


def py_contrast_focal_loss(pred,
                           target,
                           weight=None,
                           gamma=2.0,
                           alpha=0.25,
                           reduction='mean',
                           avg_factor=None,
                           contrast_gamma_scale=0.5,
                           contrast_map=None):
    """PyTorch implementation of contrast-aware sigmoid focal loss.

    Args:
        pred (Tensor): Logits, shape ``(N, C)``.
        target (Tensor): One-hot target, shape ``(N, C)``.
        weight (Tensor, optional): Sample-wise loss weight.
        gamma (float): Focusing parameter of focal loss. Default 2.0.
        alpha (float): Class-balancing parameter. Default 0.25.
        reduction (str): ``'none'`` | ``'mean'`` | ``'sum'``.
        avg_factor (int, optional): Averaging factor.
        contrast_gamma_scale (float): Maximum relative reduction of gamma at
            the most low-contrast position. Default 0.5 (i.e. effective gamma
            in ``[gamma * 0.5, gamma]``).
        contrast_map (Tensor, optional): Externally provided contrast weight
            of shape ``(N, 1)`` or ``(N,)``. If ``None``, it is estimated from
            ``pred`` via :func:`_compute_contrast_weight`.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)

    if contrast_gamma_scale > 0:
        if contrast_map is None:
            contrast_w = _compute_contrast_weight(pred)  # (N, 1) detached
        else:
            contrast_w = contrast_map.detach()
            if contrast_w.dim() == 1:
                contrast_w = contrast_w.view(-1, 1)
        # Effective gamma: smaller at low-contrast positions.
        # gamma_eff = gamma * (1 - contrast_gamma_scale * contrast_w)
        gamma_eff = gamma * (1.0 - contrast_gamma_scale * contrast_w)
        # Clamp to avoid numerical issues; pt.pow with non-int exponent is OK.
        gamma_eff = gamma_eff.clamp_min(0.0)
        focal_weight = (alpha * target + (1 - alpha) *
                        (1 - target)) * pt.pow(gamma_eff)
    else:
        focal_weight = (alpha * target + (1 - alpha) *
                        (1 - target)) * pt.pow(gamma)

    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class ContrastFocalLoss(nn.Module):
    """Contrast-Aware Focal Loss.

    A drop-in replacement for :class:`FocalLoss`. The forward signature is
    *identical*, so it can be plugged into any DETR / single-stage head
    without changing the call site.

    Args:
        use_sigmoid (bool): Must be True for now (matches FocalLoss).
        gamma (float): Focusing parameter. Default 2.0.
        alpha (float): Class-balancing parameter. Default 0.25.
        reduction (str): Default ``'mean'``.
        loss_weight (float): Scalar weight on the final loss. Default 1.0.
        activated (bool): Kept for API parity, must be False (we expect
            logits, like the standard ``FocalLoss``).
        contrast_gamma_scale (float): Strength of the contrast modulation.
            ``0.0`` reduces this loss to the standard FocalLoss. Default 0.5.
    """

    def __init__(self,
                 use_sigmoid: bool = True,
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 activated: bool = False,
                 contrast_gamma_scale: float = 0.5) -> None:
        super().__init__()
        assert use_sigmoid is True, (
            'ContrastFocalLoss only supports sigmoid mode.')
        assert activated is False, (
            'ContrastFocalLoss expects logits as input (activated=False).')
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated
        self.contrast_gamma_scale = contrast_gamma_scale

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                contrast_map=None):
        """Forward.

        Args:
            pred (Tensor): Logits, ``(N, C)``.
            target (Tensor): Class indices ``(N,)`` or one-hot ``(N, C)``.
            weight (Tensor, optional): Per-sample weight.
            avg_factor (int, optional): Averaging factor.
            reduction_override (str, optional): Override default reduction.
            contrast_map (Tensor, optional): Optional per-position contrast
                weight. When ``None``, it is auto-derived from ``pred``.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if pred.dim() != target.dim():
            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1)
            target = target[:, :num_classes]

        loss_cls = self.loss_weight * py_contrast_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor,
            contrast_gamma_scale=self.contrast_gamma_scale,
            contrast_map=contrast_map,
        )
        return loss_cls
