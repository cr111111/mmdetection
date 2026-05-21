# Dark-DINO R50 on ExDark — Stage 3 with DiffPrior enabled
#
# This config inherits from the stage-2 config and enables the
# Diffusion-Prior Feature Enhancement (DiffPrior-FPN).
# Use: load stage-2 checkpoint, then fine-tune with this config.

_base_ = ['./dark_dino_r50_exdark.py']

# Enable DiffPrior
model = dict(
    diff_prior=dict(
        type='DiffPriorFPN',
        channels=256,
        num_levels=4,
        vae_model='stabilityai/sd-vae-ft-mse',
        vae_proj_channels=64,
        num_heads=4,
        enable=True))

# Lower learning rate for fine-tuning
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00005,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
        'backbone.backbone': dict(lr_mult=0.1),
    }))

max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8],
        gamma=0.1)
]
