# Ablation: Remove Diffusion Prior (DiffPrior disabled)

_base_ = ['../dark_dino_r50_exdark_diffprior.py']

model = dict(
    diff_prior=dict(
        type='DiffPriorFPN',
        channels=256,
        num_levels=4,
        enable=False))
