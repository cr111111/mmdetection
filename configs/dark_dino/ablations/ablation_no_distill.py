# Ablation: Remove DINOv2 Distillation

_base_ = ['../dark_dino_r50_exdark_distill.py']

model = dict(distill_loss=None)
