# Ablation: Remove Retinex Decomposition (standard backbone)
#
# Uses standard ResNet-50 backbone without Retinex decomposition.
# Falls back to ChannelMapper neck (no FreqDec).

_base_ = ['../dark_dino_r50_exdark.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        _delete_=True,
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    retinex_loss=None)
