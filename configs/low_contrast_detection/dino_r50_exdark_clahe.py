# Experiment B: deterministic CLAHE enhancement followed by the same DINO.
# The model, data split, schedule, seed and geometric augmentation match A.

_base_ = ['./dino_r50_exdark_baseline.py']

_base_.train_pipeline.insert(
    1, dict(type='CLAHEEnhance', clip_limit=2.0, tile_grid_size=(8, 8)))
_base_.test_pipeline.insert(
    1, dict(type='CLAHEEnhance', clip_limit=2.0, tile_grid_size=(8, 8)))
