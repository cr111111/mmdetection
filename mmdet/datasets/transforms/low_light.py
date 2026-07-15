# Copyright (c) OpenMMLab. All rights reserved.
"""Detection-oriented transforms for low-illumination imagery."""

from typing import Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CLAHEEnhance(BaseTransform):
    """Apply CLAHE to the luminance channel of a BGR image.

    This is a deterministic, dependency-free image-enhancement baseline for
    detection experiments. It is intentionally placed before geometric
    transforms so the same enhancement is applied at train and test time.

    Args:
        clip_limit (float): Contrast threshold for limiting amplification.
        tile_grid_size (Tuple[int, int]): Size of the contextual regions.
    """

    def __init__(self,
                 clip_limit: float = 2.0,
                 tile_grid_size: Tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def transform(self, results: dict) -> dict:
        img = results['img']
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError('CLAHEEnhance expects a BGR image with 3 channels.')
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        results['img'] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(clip_limit={self.clip_limit}, '
                f'tile_grid_size={self.tile_grid_size})')
