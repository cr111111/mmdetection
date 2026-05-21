# Three-stage training schedule for Dark-DINO
#
# Stage 1 (20ep): Retinex decomposition pretraining on LOL/SICE pairs
# Stage 2 (36ep): Detection end-to-end fine-tuning with FreqDec + distill
# Stage 3 (12ep): Full fine-tuning with DiffPrior enabled

# Stage 1: Pretrain RD-Backbone
stage1_max_epochs = 20
stage1_lr = 0.0001

# Stage 2: Detection fine-tuning
stage2_max_epochs = 36
stage2_lr = 0.0001

# Stage 3: Full fine-tuning with DiffPrior
stage3_max_epochs = 12
stage3_lr = 0.00005

# Default: use stage 2 schedule (detection)
max_epochs = stage2_max_epochs

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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
