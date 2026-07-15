# Experiment C: detection-aware Dark-DINO on ExDark.
# It keeps the same data split, schedule and optimizer as Experiments A/B.

_base_ = [
    '../dark_dino/_base_/dark_dino_r50.py',
    './_base_/exdark_common.py',
]

model = dict(bbox_head=dict(num_classes=12))
