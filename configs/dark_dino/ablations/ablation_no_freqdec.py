# Ablation: Remove Frequency-Decoupled Neck (use ChannelMapper)

_base_ = ['../dark_dino_r50_exdark.py']

model = dict(
    neck=dict(
        _delete_=True,
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4))
