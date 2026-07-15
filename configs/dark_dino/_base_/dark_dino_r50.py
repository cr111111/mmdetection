# Dark-DINO with R50 backbone 鈥?shared model skeleton
# This is the base config that other dark_dino configs inherit from.

num_levels = 4
model = dict(
    type='DarkDINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    backbone=dict(
        type='RetinexDecomposedBackbone',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
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
        freeze_decomposer=False,
        pp_mean=[123.675, 116.28, 103.53],
        pp_std=[58.395, 57.12, 57.375]),
    neck=dict(
        type='FreqDecoupledNeck',
        in_channels=[512, 1024, 2048],
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
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    bbox_head=dict(
        type='DINOHead',
        num_classes=80,  # overridden by dataset config
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    # Retinex consistency loss
    retinex_loss=dict(
        type='RetinexConsistencyLoss',
        recon_weight=1.0,
        smooth_weight=0.5,
        color_weight=0.1),
    # Feature distillation loss (disabled by default, enabled in stage 2)
    distill_loss=None,
    # Diffusion prior (disabled by default, enabled in stage 3)
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
