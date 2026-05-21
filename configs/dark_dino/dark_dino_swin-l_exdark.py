# Dark-DINO Swin-L on ExDark — large model version for SOTA comparison
#
# This config uses Swin-Large backbone for maximum accuracy.
# Requires ~65GB VRAM per card with bs=2 on 2×A800.

_base_ = [
    './_base_/exdark_detection.py',
    '../../_base_/default_runtime.py',
]

num_levels = 5
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'

model = dict(
    type='DarkDINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    backbone=dict(
        type='RetinexDecomposedBackbone',
        backbone=dict(
            type='SwinTransformer',
            pretrain_img_size=384,
            embed_dims=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=True,
            convert_weights=True,
            init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
        decomposer=dict(
            type='RetinexDecomposer',
            in_channels=3,
            mid_channels=32,
            num_layers=3),
        light_encoder=dict(
            type='IlluminationTokenEncoder',
            in_channels=3,
            out_channels=256,
            num_tokens=4),
        freeze_backbone_stage1=True),
    neck=dict(
        type='FreqDecoupledNeck',
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        kernel_size=1,
        num_outs=num_levels,
        low_ratio=0.25,
        high_ratio=0.75,
        use_mamba=True,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32)),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_levels=num_levels, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=256, num_levels=num_levels, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='DINOHead',
        num_classes=12,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    retinex_loss=dict(
        type='RetinexConsistencyLoss',
        recon_weight=1.0,
        smooth_weight=0.5,
        color_weight=0.1),
    distill_loss=None,
    diff_prior=None,
    freq_gate_reg_weight=0.01,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)])),
    test_cfg=dict(max_per_img=300))

# ExDark dataset
data_root = 'data/exdark/'

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/exdark_train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline={{_base_.train_pipeline}},
        metainfo=dict(classes=(
            'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair',
            'Cup', 'Dog', 'Motorbike', 'People', 'Table'))))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/exdark_val.json',
        data_prefix=dict(img='images/'),
        pipeline={{_base_.test_pipeline}},
        metainfo=dict(classes=(
            'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair',
            'Cup', 'Dog', 'Motorbike', 'People', 'Table'))))

test_dataloader = {{_base_.val_dataloader}}

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/exdark_val.json',
    metric='bbox')
test_evaluator = {{_base_.val_evaluator}}

max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
        'backbone.backbone': dict(lr_mult=0.1),
    }))

auto_scale_lr = dict(base_batch_size=2)
