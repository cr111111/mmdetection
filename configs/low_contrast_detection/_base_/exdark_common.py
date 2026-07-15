# Shared ExDark setup for controlled low-contrast detection experiments.

_base_ = ['../../_base_/default_runtime.py']

data_root = 'data/exdark/'
classes = (
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog',
    'Motorbike', 'People', 'Table')

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ], [
            dict(
                type='RandomChoiceResize',
                scales=[(400, 4200), (500, 4200), (600, 4200)],
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(384, 600),
                allow_negative_crop=True),
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ]]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

metainfo = dict(classes=classes)
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
        metainfo=metainfo,
        pipeline=train_pipeline))
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
        metainfo=metainfo,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric', ann_file=data_root + 'annotations/exdark_val.json',
    metric='bbox')
test_evaluator = val_evaluator

max_epochs = 36
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True,
        milestones=[30], gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))
auto_scale_lr = dict(base_batch_size=4)
randomness = dict(seed=3407, deterministic=False)

# Keep exactly one best validation checkpoint for automated final evaluation.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, save_best='coco/bbox_mAP',
        rule='greater', max_keep_ckpts=2))
