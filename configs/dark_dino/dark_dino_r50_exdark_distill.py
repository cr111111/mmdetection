# Dark-DINO R50 on ExDark — with DINOv2 Distillation enabled
#
# This config enables the Cross-Domain DINOv2 Distillation (CD-Distill).
# Requires paired dark/normal images in the training dataset.

_base_ = ['./dark_dino_r50_exdark.py']

# Enable Distillation
model = dict(
    distill_loss=dict(
        type='FeatureDistillationLoss',
        cosine_weight=1.0,
        relation_weight=0.5,
        temperature=4.0,
        num_levels=4,
        student_channels=256,
        teacher_channels=1024,
        teacher_cfg=dict(
            model_name='dinov2_vitl14',
            output_layers=[8, 16, 20, 24])))
