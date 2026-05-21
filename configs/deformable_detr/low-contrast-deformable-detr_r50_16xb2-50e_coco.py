"""Low-contrast Deformable DETR (R50, 16xb2, 50e, COCO).

This config wires together the three innovations introduced for low-contrast
object detection:

* ``ContrastAwareNeck`` instead of ``ChannelMapper``
* ``ContrastFocalLoss`` instead of ``FocalLoss`` for the classification head
* ``LowContrastDeformableDETR`` instead of ``DeformableDETR`` so that the
  decoder reference points are partly initialised at low-variance positions
  of the encoder memory.

It inherits the vanilla Deformable DETR config and only overrides the three
fields above plus a new ``low_contrast_ratio`` argument on the detector.
"""

_base_ = './deformable-detr_r50_16xb2-50e_coco.py'

model = dict(
    type='LowContrastDeformableDETR',
    # Fraction of queries whose initial reference points are placed on
    # low-variance (low-contrast) positions of the encoder memory.
    low_contrast_ratio=0.3,
    # Use the highest-resolution feature level (index 0) for selecting the
    # low-variance positions; small / low-contrast targets benefit most
    # from it.
    low_contrast_level=0,
    neck=dict(
        _delete_=True,
        type='ContrastAwareNeck',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4,
        local_contrast_kernel=7,
        contrast_scale=1.0,
        use_cross_scale_se=True),
    bbox_head=dict(
        loss_cls=dict(
            _delete_=True,
            type='ContrastFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
            contrast_gamma_scale=0.5)))
