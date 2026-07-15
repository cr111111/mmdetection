"""Unit tests for Dark-DINO corrected components.

Uses importlib to load modules directly from file paths, bypassing the
mmdet package __init__.py chain (which requires compiled mmcv._ext).

Run with:
    python tests/test_dark_dino_fixes.py
"""

import sys
import os
import importlib.util
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJ_ROOT = os.path.join(os.path.dirname(__file__), '..')

# ---------------------------------------------------------------------------
# Build a complete stub package tree so that `from mmdet.xxx import yyy`
# and `from ..utils.zzz import www` work without triggering mmdet.__init__
# (which pulls in mmcv._ext).
# ---------------------------------------------------------------------------

# 1. Registry stub
class _RegistryStub:
    def register_module(self, module=None, force=False, name=None):
        if module is not None:
            return module
        def decorator(cls):
            return cls
        return decorator


# 2. Build fake package hierarchy
def _make_pkg(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
    return mod

mmdet_pkg = _make_pkg('mmdet')
models_pkg = _make_pkg('mmdet.models')
backbones_pkg = _make_pkg('mmdet.models.backbones')
necks_pkg = _make_pkg('mmdet.models.necks')
losses_pkg = _make_pkg('mmdet.models.losses')
utils_pkg = _make_pkg('mmdet.models.utils')

# Link parent references
mmdet_pkg.models = models_pkg
models_pkg.backbones = backbones_pkg
models_pkg.necks = necks_pkg
models_pkg.losses = losses_pkg
models_pkg.utils = utils_pkg

# 3. Registry module
reg_mod = types.ModuleType('mmdet.registry')
reg_mod.MODELS = _RegistryStub()
sys.modules['mmdet.registry'] = reg_mod
mmdet_pkg.registry = reg_mod

# 4. mmdet.utils stub with type aliases
utils_top = types.ModuleType('mmdet.utils')
utils_top.OptConfigType = type(None)
utils_top.OptMultiConfig = type(None)
utils_top.ConfigType = dict
sys.modules['mmdet.utils'] = utils_top
mmdet_pkg.utils = utils_top

# 5. mmengine (real if available, otherwise stub)
try:
    from mmengine.model import BaseModule
    from mmengine.config import ConfigDict
except ImportError:
    class BaseModule(nn.Module):
        def __init__(self, init_cfg=None):
            super().__init__()
            self.init_cfg = init_cfg
        def init_weights(self):
            pass

# 6. mmcv.cnn.ConvModule stub (mmcv is installed but ops._ext fails;
#    we only need ConvModule which is pure Python)
try:
    from mmcv.cnn import ConvModule
except ImportError:
    class ConvModule(nn.Module):
        """Minimal ConvModule stub: Conv2d + optional Norm + optional Act."""
        def __init__(self, in_channels, out_channels, kernel_size,
                     stride=1, padding=0, dilation=1, groups=1, bias=False,
                     conv_cfg=None, norm_cfg=None, act_cfg=None, **kwargs):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
            self.norm = None
            if norm_cfg and norm_cfg.get('type') == 'GN':
                self.norm = nn.GroupNorm(
                    norm_cfg.get('num_groups', 32), out_channels)
            elif norm_cfg and norm_cfg.get('type') == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            self.act = None
            if act_cfg and act_cfg.get('type') in ('ReLU', 'GELU', 'SiLU'):
                act_type = act_cfg['type']
                if act_type == 'ReLU':
                    self.act = nn.ReLU(inplace=True)
                elif act_type == 'GELU':
                    self.act = nn.GELU()
                elif act_type == 'SiLU':
                    self.act = nn.SiLU()

        def forward(self, x):
            x = self.conv(x)
            if self.norm is not None:
                x = self.norm(x)
            if self.act is not None:
                x = self.act(x)
            return x

    # Inject into mmcv.cnn namespace
    mmcv_cnn = types.ModuleType('mmcv.cnn')
    mmcv_cnn.ConvModule = ConvModule
    sys.modules['mmcv.cnn'] = mmcv_cnn
    if 'mmcv' not in sys.modules:
        sys.modules['mmcv'] = types.ModuleType('mmcv')
    sys.modules['mmcv'].cnn = mmcv_cnn


# ---------------------------------------------------------------------------
# Load target modules
# ---------------------------------------------------------------------------
def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load utils first (no internal deps)
dct_utils = _load_module(
    'mmdet.models.utils.dct_utils',
    os.path.join(PROJ_ROOT, 'mmdet', 'models', 'utils', 'dct_utils.py'))
mamba_block = _load_module(
    'mmdet.models.utils.mamba_block',
    os.path.join(PROJ_ROOT, 'mmdet', 'models', 'utils', 'mamba_block.py'))

# Register in package namespace for relative import resolution
utils_pkg.dct_utils = dct_utils
utils_pkg.mamba_block = mamba_block

# Load backbone, neck, loss
backbone_mod = _load_module(
    'mmdet.models.backbones.retinex_decomposed_backbone',
    os.path.join(PROJ_ROOT, 'mmdet', 'models', 'backbones',
                 'retinex_decomposed_backbone.py'))
neck_mod = _load_module(
    'mmdet.models.necks.freq_decoupled_neck',
    os.path.join(PROJ_ROOT, 'mmdet', 'models', 'necks',
                 'freq_decoupled_neck.py'))
loss_mod = _load_module(
    'mmdet.models.losses.retinex_consistency_loss',
    os.path.join(PROJ_ROOT, 'mmdet', 'models', 'losses',
                 'retinex_consistency_loss.py'))

# Register in package namespaces
backbones_pkg.retinex_decomposed_backbone = backbone_mod
necks_pkg.freq_decoupled_neck = neck_mod
losses_pkg.retinex_consistency_loss = loss_mod


# ---------------------------------------------------------------------------
# Convenient aliases
# ---------------------------------------------------------------------------
dct2 = dct_utils.dct2
idct2 = dct_utils.idct2
freq_band_masks = dct_utils.freq_band_masks
split_freq_bands = dct_utils.split_freq_bands

MambaS6Block = mamba_block.MambaS6Block
RetinexDecomposer = backbone_mod.RetinexDecomposer
IlluminationTokenEncoder = backbone_mod.IlluminationTokenEncoder
RetinexDecomposedBackbone = backbone_mod.RetinexDecomposedBackbone
FreqDecoupledNeck = neck_mod.FreqDecoupledNeck
MidBandSelfAttention = neck_mod.MidBandSelfAttention
RetinexConsistencyLoss = loss_mod.RetinexConsistencyLoss


# ---------------------------------------------------------------------------
# Minimal ResNet stub for backbone test (avoids importing real ResNet
# which needs mmcv._ext)
# ---------------------------------------------------------------------------
class SimpleBackbone(nn.Module):
    """3-layer conv backbone that returns 3 feature levels."""
    def __init__(self, **kwargs):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU())

    def forward(self, x):
        c0 = self.layer0(x)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        return (c1, c2, c3)


# Monkey-patch MODELS.build to return SimpleBackbone for ResNet configs
_original_build = _RegistryStub.register_module

def _patched_build(self, cfg, **kwargs):
    if isinstance(cfg, dict) and cfg.get('type') == 'ResNet':
        return SimpleBackbone(**{k: v for k, v in cfg.items()
                                  if k != 'type'})
    return None

_RegistryStub.build = _patched_build


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_dct_round_trip():
    """IDCT(DCT(x)) should recover x."""
    print("=== Test 1: DCT round-trip ===")
    x = torch.randn(2, 4, 32, 48)
    x_recon = idct2(dct2(x))
    err = (x - x_recon).abs().max().item()
    print(f"  Max recon error: {err:.2e}")
    assert err < 1e-4, f"DCT round-trip error too large: {err}"
    print("  PASS\n")


def test_freq_mask_partition():
    """Low + mid + high = 1 everywhere."""
    print("=== Test 2: Mask partition of unity ===")
    low, mid, high = freq_band_masks(32, 48, 0.25, 0.75)
    err = (low + mid + high - 1.0).abs().max().item()
    print(f"  Max partition error: {err:.2e}")
    assert err < 1e-4, f"Mask partition error too large: {err}"
    assert low.min() >= 0 and low.max() <= 1
    assert mid.min() >= 0 and mid.max() <= 1
    assert high.min() >= 0 and high.max() <= 1
    print("  PASS\n")


def test_freq_mask_smooth():
    """Masks should have smooth (non-binary) transitions."""
    print("=== Test 3: Masks are smooth ===")
    low, mid, high = freq_band_masks(64, 64, 0.25, 0.75, smooth_width=0.05)
    assert ((low > 0.01) & (low < 0.99)).any().item()
    assert ((mid > 0.01) & (mid < 0.99)).any().item()
    assert ((high > 0.01) & (high < 0.99)).any().item()
    print("  PASS\n")


def test_split_freq_bands():
    """split_freq_bands: shapes + sum ~ input."""
    print("=== Test 4: split_freq_bands ===")
    x = torch.randn(2, 8, 32, 32)
    low, mid, high = split_freq_bands(x, 0.25, 0.75, smooth_width=0.05)
    assert low.shape == x.shape
    err = (low + mid + high - x).abs().max().item()
    print(f"  Band sum error: {err:.2e}")
    assert err < 1e-4
    print("  PASS\n")


def test_retinex_decomposer():
    """RetinexDecomposer: shapes + range + gradients."""
    print("=== Test 5: RetinexDecomposer ===")
    decomposer = RetinexDecomposer(
        in_channels=3, mid_channels=32, num_layers=3)
    x = torch.rand(2, 3, 64, 80)
    R, L = decomposer(x)
    assert R.shape == x.shape and L.shape == x.shape
    assert R.min() >= 0 and R.max() <= 1, \
        f"R out of [0,1]: [{R.min()}, {R.max()}]"
    assert L.min() >= 0 and L.max() <= 1, \
        f"L out of [0,1]: [{L.min()}, {L.max()}]"
    print(f"  R:[{R.min():.4f},{R.max():.4f}] L:[{L.min():.4f},{L.max():.4f}]")
    (R.sum() + L.sum()).backward()
    assert all(p.grad is not None
               for p in decomposer.parameters() if p.requires_grad)
    print("  PASS\n")


def test_retinex_backbone_denorm():
    """RetinexDecomposedBackbone: denorm pipeline."""
    print("=== Test 6: RetinexDecomposedBackbone denorm ===")
    backbone = RetinexDecomposedBackbone(
        backbone=dict(
            type='ResNet', depth=18, num_stages=4, out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True, style='pytorch'),
        decomposer=dict(in_channels=3, mid_channels=16, num_layers=2),
        light_encoder=dict(in_channels=3, out_channels=256, num_tokens=4),
        pp_mean=[123.675, 116.28, 103.53],
        pp_std=[58.395, 57.12, 57.375])

    # Simulate DetDataPreprocessor output
    x_raw = torch.rand(1, 3, 128, 160) * 255.0
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
    x_norm = (x_raw - mean) / std

    mlvl_feats, light_tokens, R, L = backbone(x_norm)
    assert isinstance(mlvl_feats, tuple)
    assert len(mlvl_feats) == 3
    assert light_tokens.shape == (1, 4, 256)
    assert R.shape == (1, 3, 128, 160)
    assert R.min() >= 0 and R.max() <= 1, \
        f"R out of [0,1]: [{R.min()}, {R.max()}]"
    assert L.min() >= 0 and L.max() <= 1, \
        f"L out of [0,1]: [{L.min()}, {L.max()}]"
    print(f"  levels={len(mlvl_feats)} R:[{R.min():.4f},{R.max():.4f}]")
    loss = sum(f.sum() for f in mlvl_feats) + R.sum()
    loss.backward()
    print("  PASS\n")


def test_freq_neck():
    """FreqDecoupledNeck: forward + gradients."""
    print("=== Test 7: FreqDecoupledNeck ===")
    neck = FreqDecoupledNeck(
        in_channels=[128, 256, 512],
        out_channels=64,
        kernel_size=1,
        num_outs=4,
        low_ratio=0.25, high_ratio=0.75,
        smooth_width=0.05,
        use_mamba=True)
    feats = (
        torch.randn(2, 128, 32, 40),
        torch.randn(2, 256, 16, 20),
        torch.randn(2, 512, 8, 10),
    )
    light_map = torch.rand(2, 3, 32, 40)
    out = neck(feats, light_map=light_map)
    assert len(out) == 4
    for i, o in enumerate(out):
        assert o.shape[0] == 2 and o.shape[1] == 64
    sum(o.sum() for o in out).backward()
    assert all(p.grad is not None
               for p in neck.parameters() if p.requires_grad)
    print(f"  Output: {len(out)} levels, {out[0].shape[1]} ch")
    print("  PASS\n")


def test_mamba():
    """MambaS6Block: forward + backward + large seq."""
    print("=== Test 8: MambaS6Block ===")
    block = MambaS6Block(
        channels=64, d_state=8, d_conv=3, expand_ratio=2,
        chunk_size=32, bidirectional=True, max_seq_len=512)
    x = torch.randn(2, 100, 64, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    print(f"  Grad norm: {x.grad.norm().item():.4f}")

    x2 = torch.randn(1, 8000, 64, requires_grad=True)
    y2 = block(x2)
    assert y2.shape == x2.shape
    print("  Large seq (N=8000): OK")
    print("  PASS\n")


def test_retinex_loss():
    """RetinexConsistencyLoss: forward + finite grads."""
    print("=== Test 9: RetinexConsistencyLoss ===")
    loss_fn = RetinexConsistencyLoss(
        recon_weight=1.0, smooth_weight=0.5, color_weight=0.1)
    I = torch.rand(2, 3, 64, 80, requires_grad=True)
    R = torch.rand(2, 3, 64, 80, requires_grad=True)
    L = torch.rand(2, 3, 64, 80, requires_grad=True)
    losses = loss_fn(I, R, L)
    assert 'loss_recon' in losses
    assert 'loss_smooth' in losses
    assert 'loss_color' in losses
    sum(losses.values()).backward()
    for n, v in [("I", I), ("R", R), ("L", L)]:
        assert v.grad is not None and torch.isfinite(v.grad).all()
    print(f"  recon={losses['loss_recon'].item():.4f} "
          f"smooth={losses['loss_smooth'].item():.4f} "
          f"color={losses['loss_color'].item():.4f}")
    print("  PASS\n")


def test_naming():
    """Verify DeformableAttnBlock renamed."""
    print("=== Test 10: MidBandSelfAttention naming ===")
    assert not hasattr(neck_mod, 'DeformableAttnBlock'), \
        "DeformableAttnBlock should be renamed"
    block = MidBandSelfAttention(channels=64, num_heads=4)
    x = torch.randn(2, 64, 16, 20)
    assert block(x).shape == x.shape
    print("  PASS\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Dark-DINO Fix Verification Tests")
    print("=" * 60 + "\n")

    test_dct_round_trip()
    test_freq_mask_partition()
    test_freq_mask_smooth()
    test_split_freq_bands()
    test_retinex_decomposer()
    test_retinex_backbone_denorm()
    test_freq_neck()
    test_mamba()
    test_retinex_loss()
    test_naming()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
