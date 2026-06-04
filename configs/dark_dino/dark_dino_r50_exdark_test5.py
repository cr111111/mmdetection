# Dark-DINO R50 on ExDark — Quick Test 5 Epochs
# ==================================================
# Full model: RD-Backbone + FreqDec-Neck + DINOv2 Distill + DiffPrior-FPN
# For quick validation before full training runs.
#
# Usage:
#   mmdet train configs/dark_dino/dark_dino_r50_exdark_test5.py \
#            --work-dir work_dirs/dark_dino_exdark_test5

_base_ = [
    './_base_/dark_dino_r50.py',
    './_base_/exdark_detection.py',
    '../_base_/default_runtime.py',
]

# ---- Dataset ----
data_root = 'data/exdark/'

# ExDark 12 categories
CLASSES = (
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair',
    'Cup', 'Dog', 'Motorbike', 'People', 'Table')

# Override num_classes
model = dict(
    bbox_head=dict(num_classes=12),
    # ===== Innovation #1: Retinex Decomposition Loss (always on) =====
    retinex_loss=dict(
        type='RetinexConsistencyLoss',
        recon_weight=1.0,
        smooth_weight=0.5,
        color_weight=0.1),
    # ===== Innovation #3: Cross-Domain DINOv2 Distillation =====
    # _delete_=True required because base sets distill_loss=None
    distill_loss=dict(
        _delete_=True,
        type='FeatureDistillationLoss',
        cosine_weight=1.0,
        relation_weight=0.5,
        temperature=4.0,
        num_levels=4,
        student_channels=256,
        teacher_channels=1024,
        teacher_cfg=dict(
            model_name='dinov2_vitl14',
            output_layers=[8, 16, 20, 24])),
    # ===== Innovation #4: Diffusion-Prior Feature Enhancement =====
    # _delete_=True required because base sets diff_prior=None
    diff_prior=dict(
        _delete_=True,
        type='DiffPriorFPN',
        channels=256,
        num_levels=4,
        vae_model='stabilityai/sd-vae-ft-mse',
        vae_proj_channels=64,
        num_heads=4,
        enable=True),
)

# ---- Dataloaders ----
train_dataloader = dict(
    batch_size=2,
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
        metainfo=dict(classes=CLASSES)))

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
        metainfo=dict(classes=CLASSES)))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/exdark_val.json',
    metric='bbox')
test_evaluator = val_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ---- Schedule: 5 epochs for quick test ----
max_epochs = 5
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)   # validate every epoch

# Linear warmup for first epoch, then constant-decay
param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=1,          # warmup over 1 epoch
        start_factor=0.01,
        by_epoch=True),
    dict(
        type='MultiStepLR',
        begin=1,
        end=max_epochs,
        by_epoch=True,
        milestones=[3],  # drop LR at epoch 3
        gamma=0.1)
]

# Optimizer with differential LR for backbone
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,       # base learning rate
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),      # backbone at 10% LR
        'backbone.backbone': dict(lr_mult=0.1),
        'diff_prior': dict(lr_mult=0.5),     # DiffPrior at 50% LR
    }))

auto_scale_lr = dict(base_batch_size=4)

# ---- Logging & Checkpointing (aggressive for short run) ----
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,           # save every epoch (only 5 total)
        save_best='auto',     # save best by mAP
        max_keep_ckpts=3,     # keep last 3 checkpoints
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
