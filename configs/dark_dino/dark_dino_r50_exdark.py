# Dark-DINO R50 on ExDark — Stage 2 (main detection config)
#
# This is the primary config for detection training on ExDark.
# It includes RD-Backbone + FreqDec-Neck + DINO decoder.
# Distillation and DiffPrior are disabled by default (enable via overrides).

_base_ = [
    './_base_/dark_dino_r50.py',
    './_base_/exdark_detection.py',
    '../../_base_/default_runtime.py',
]

# ExDark dataset config (12 categories)
# NOTE: Replace data_root with your actual ExDark COCO-format path
data_root = 'data/exdark/'

# Override num_classes for ExDark (12 categories)
model = dict(
    bbox_head=dict(num_classes=12),
)

# Dataset
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

# Evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/exdark_val.json',
    metric='bbox')
test_evaluator = {{_base_.val_evaluator}}

# Training schedule
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

auto_scale_lr = dict(base_batch_size=4)
