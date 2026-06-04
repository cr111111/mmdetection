# Dark-DINO 完整实验指南

> 本文档对应论文 `paper.md` 中所有表格的实验操作指南。完成所有实验后，将结果直接填入论文表格。

---

## 一、环境准备

### 1.1 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | 2× A100/A800 (40G+) | 2× A800 (80G) |
| 内存 | 64 GB | 128 GB |
| 硬盘 | 200 GB SSD | 500 GB NVMe |

### 1.2 软件依赖

```bash
# 核心依赖（mmdet 已安装基础上）
pip install diffusers transformers accelerate
pip install mamba-ssm          # Mamba SSM
pip install timm               # DINOv2

# 预训练模型下载
# SD VAE: stabilityai/sd-vae-ft-mse (~167MB, 自动下载)
# DINOv2: facebook/dinov2-vitl14 (~1.2GB, 自动下载)
```

---

## 二、数据集准备

### 2.1 ExDark（主基准）

**用途**: Stage 2/3 检测训练 + 所有消融实验 + 主结果 Table 1/2

```bash
# 目录结构
data/exdark/
├── images/
│   ├── train/        # 低光训练图像
│   └── val/          # 低光验证图像
└── annotations/
    ├── exdark_train.json   # COCO格式标注 (12类)
    └── exdark_val.json     # COCO格式标注

# 12个类别:
# Bicycle, Boat, Bottle, Bus, Car, Cat,
# Chair, Cup, Dog, Motorbike, People, Table
```

**下载**: ExDark 官网或 Kaggle。需自行转换为 COCO 格式标注。

**注意**: 配置中 `data_root = 'data/exdark/'`，请根据实际路径修改各 `.py` 配置中的 `data_root`。

### 2.2 LOL（配对低光/正常光）

**用途**: Stage 1 Retinex 分解预训练 + 蒸馏教师输入

```bash
data/lol/
├── low/              # 低光图像 (训练集)
├── high/             # 对应的正常光图像 (蒸馏用)
└── annotations/
    ├── train.json    # COCO格式 (可无bbox, 仅需img_id映射)
    └── val.json
```

**来源**: [LOL Dataset](https://github.com/weixiong-ur/mdr-net) (485 train / 15 test)

**关键**: 蒸馏需要配对的正常光图像。LOL 的 `high/` 文件夹即为配对正常光图。

### 2.3 DarkFace（跨数据集泛化）

**用途**: Table 2 泛化评估

```bash
data/darkface/
├── images/
└── annotations/
    ├── darkface_test.json   # COCO格式 (仅人脸类别)
```

**来源**: [DarkFace Dataset](https://github.com/yang-fengbeibei/DarkFace)

### 2.4 数据集与论文表格对照关系

| 论文表格 | 数据集 | 用途 |
|---------|--------|------|
| **Table 1** | ExDark val | 主结果 SOTA 对比 |
| **Table 2** | DarkFace test | 跨数据集泛化 |
| **Table 3-7** | ExDark val | 全部消融实验 |
| **Fig. 3-6** | ExDark val | 可视化分析 |

---

## 三、三阶段训练流程（核心）

### 总体流程图

```
Stage 1 (20ep)          Stage 2 (36ep)            Stage 3 (12ep)
┌─────────────┐    →    ┌─────────────┐      →    ┌─────────────┐
│ LOL/SICE    │         │ ExDark       │           │ ExDark       │
│ 仅分解器训练 │         │ 冻结分解器    │           │ 解冻全部      │
│ LR=2e-4     │         │ + FreqDec    │           │ + DiffPrior   │
│ bs=4        │         │ + Distill    │           │ LR=5e-5      │
└─────────────┘         │ LR=1e-4      │           └─────────────┘
 checkpoint → load →     │ bs=2        │           checkpoint → load →
                         └─────────────┘
```

### Stage 1: Retinex 分解预训练（20 epochs）

**目的**: 让 Retinex 分解器学会物理上有意义的 R/L 分解

```bash
cd d:/my_project/mmdetection

# 单卡或多卡皆可
python tools/train.py \
    configs/dark_dino/stage1_pretrain_rd.py \
    --work-dir work_dirs/dark_dino_stage1 \
    [ --gpu-ids 0 1 ]   # 多卡 DDP
```

**Stage 1 关键参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | LOL low/ | 配对低光图像 |
| Epochs | 20 | 验证间隔=2 |
| LR | 2e-4 | 分解器专用 |
| Backbone LR mult | 0.0 | **完全冻结** ResNet |
| Batch size | 4 (per GPU) | 2卡共8 |
| λ_recon | **10.0** | 比检测时高10倍（专注重建） |
| λ_smooth | 1.0 | 光照平滑约束 |
| λ_color | 0.5 | 灰色世界先验 |
| LR schedule | MultiStep[15] | 第15轮衰减 |
| 输出 | `work_dirs/dark_dino_stage1/epoch_20.pth` | ← Stage 2 加载此权重 |

**Stage 1 完成标志**: reconstruction loss 收敛到 < 0.05，可视化 R/L 分解质量合理。

---

### Stage 2: 检测端到端微调（36 epochs）

**目的**: 在冻结好的分解器上训练完整检测流水线 + 蒸馏

#### 2a: 不含 DiffPrior 的基础版本（主实验）

```bash
python tools/train.py \
    configs/dark_dino/dark_dino_r50_exdark.py \
    --work-dir work_dirs/dark_dino_stage2_base \
    --resume work_dirs/dark_dino_stage1/epoch_20.pth \  # 加载 Stage 1 权重
    --gpu-ids 0 1
```

#### 2b: 含 CD-Distill 的版本（需要配对正常光）

```bash
# 此版本同时启用蒸馏，需确保数据集包含 paired normal-light 图像路径
python tools/train.py \
    configs/dark_dino/dark_dino_r50_exdark_distill.py \
    --work-dir work_dirs/dark_dino_stage2_distill \
    --resume work_dirs/dark_dino_stage1/epoch_20.pth \
    --gpu-ids 0 1
```

**Stage 2 关键参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | ExDark train | 低光目标检测 |
| Epochs | 36 | 验证间隔=1 |
| LR | 1e-4 | AdamW |
| Backbone LR mult | 0.1 | backbone=1e-5 |
| Batch size | 2 (per GPU) | 2卡共4 |
| λ_recon | 1.0 | 保持分解（较低权重） |
| λ_smooth | 0.5 | |
| λ_color | 0.1 | |
| λ_cosine | 1.0 | **蒸馏启用** |
| λ_relation | 0.5 | **蒸馏启用** |
| λ_gate | 0.01 | 门控正则 |
| LR schedule | MultiStep[30] | 第30轮衰减 |
| 冻结分解器 | 是 (`freeze_decomposer=True`) | 防止被检测梯度破坏 |
| 输出 | `work_dirs/dark_dino_stage2_*/epoch_36.pth` | ← Stage 3 加载 |

---

### Stage 3: Diffusion Prior 微调（12 epochs）

**目的**: 融合 SD VAE 先验，全量微调

```bash
python tools/train.py \
    configs/dark_dino/dark_dino_r50_exdark_diffprior.py \
    --work-dir work_dirs/dark_dino_stage3_full \
    --resume work_dirs/dark_dino_stage2_distill/epoch_36.pth \  # 加载 Stage 2 最佳
    --gpu-ids 0 1
```

**Stage 3 关键参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | ExDark train | 同 Stage 2 |
| Epochs | 12 | 验证间隔=1 |
| LR | **5e-5** | Stage 2 的一半 |
| DiffPrior | **启用** | VAE encoder forward |
| SD VAE | frozen | 不更新 VAE 参数 |
| LR schedule | MultiStep[8] | 第8轮衰减 |
| 输出 | `work_dirs/dark_dino_stage3_full/epoch_12.pth` | ← 最终模型 |

---

## 四、对照实验（SOTA Comparison）

### 4.1 Table 1: ExDark 主结果

**对应论文**: Section 4.3, Table 1

**基线方法**（需复现/查找 reported numbers）:

| 方法 | 来源 | 配置说明 |
|------|------|---------|
| Faster R-CNN [1] | mmdet 官方 | `configs/faster_rcnn/` on ExDark |
| DINO [3] | mmdet 官方 | `configs/dino/` on ExDark（我们的 baseline） |
| MAET [14] | ECCV 2022 | 复现 or 引用原论文 reported |
| IAT+DINO [15] | ICCV 2023 | IAT 增强 → DINO 检测 |
| FeatEnHancer+DINO [17] | CVPR 2023 | 特征增强 → DINO 检测 |
| DAI-Net [18] | NeurIPS 2023 | 域自适应方法 |

**我们的方法**:

| 配置文件 | 对应行 | 训练方式 |
|----------|--------|---------|
| `dark_dino_r50_exdark.py` | Dark-DINO (R50) | Stage 1→2→3 完整训练 |
| `dark_dino_swin-l_exdark.py` | Dark-DINO (Swin-L) | 同上，Swin-L backbone |

**运行命令（完整版）**:

```bash
# ===== 完整三阶段训练脚本（R50 版本）=====

# Stage 1
python tools/train.py configs/dark_dino/stage1_pretrain_rd.py \
    --work-dir work_dirs/exp_main/stage1 --gpu-ids 0 1

# Stage 2 (base, 无蒸馏无DiffPrior)
python tools/train.py configs/dark_dino/dark_dino_r50_exdark.py \
    --work-dir work_dirs/exp_main/stage2_base \
    --resume work_dirs/exp_main/stage1/epoch_20.pth \
    --gpu-ids 0 1

# Stage 2 (with distill)
python tools/train.py configs/dark_dino/dark_dino_r50_exdark_distill.py \
    --work-dir work_dirs/exp_main/stage2_distill \
    --resume work_dirs/exp_main/stage1/epoch_20.pth \
    --gpu-ids 0 1

# Stage 3 (full model with DiffPrior)
python tools/train.py configs/dark_dino/dark_dino_r50_exdark_diffprior.py \
    --work-dir work_dirs/exp_main/stage3_full \
    --resume work_dirs/exp_main/stage2_distill/epoch_36.pth \
    --gpu-ids 0 1

# ===== Swin-L 版本（同样三阶段，替换 config）=====
python tools/train.py configs/dark_dino/dark_dino_swin-l_exdark.py \
    --work-dir work_dirs/exp_main/swinl_stage2 \
    --gpu-ids 0 1   # 注意 bs=1, 可能需要梯度累积
```

**评估命令**:

```bash
# 评估最佳 checkpoint（通常取 bbox_mAP 最高的 epoch）
python tools/test.py \
    configs/dark_dino/dark_dino_r50_exdark_diffprior.py \
    work_dirs/exp_main/stage3_full/epoch_12.pth \
    --work-dir work_dirs/exp_main/eval_final
```

**输出指标**: AP, AP50, AP75, APS, APM, APL （COCO-style）

---

### 4.2 Table 2: DarkFace 跨数据集泛化

**对应论文**: Section 4.3, Table 2

**操作**: 用 ExDark 训练好的最终模型直接在 DarkFace 上测试（不 fine-tune）。

```bash
# 需要准备 DarkFace 的测试配置（类似 exdark_detection.py 但指向 darkface 路径）
# 或临时修改 data_root 和 ann_file

python tools/test.py \
    configs/dark_dino/dark_dino_r50_exdark_diffprior.py \
    work_dirs/exp_main/stage3_full/epoch_12.pth \
    --work-dir work_dirs/exp_main/eval_darkface \
    --cfg-options \
        data_root='data/darkface/' \
        val_dataloader.dataset.ann_file='annotations/darkface_test.json' \
        val_evaluator.ann_file='data/darkface/annotations/darkface_test.json' \
        model.bbox_head.num_classes=1
```

---

## 五、消融实验（Ablation Studies）

### 5.1 Table 3: 组件级消融 ✅ 配置就绪

**对应论文**: Section 4.4, Table 3 — **最核心的消融表**

**目的**: 验证每个模块的独立贡献（加法性）

| 实验 ID | RD-Backbone | FreqDec | DiffPrior | Distill | 配置文件 | 对应行 |
|---------|:-----------:|:-------:|:---------:|:-------:|----------|--------|
| Baseline | ✗ | ✗ | ✗ | ✗ | 标准 DINO R50 on ExDark | Row 1 |
| +RD | ✓ | ✗ | ✗ | ✗ | `ablations/ablation_no_freqdec.py` + 移除FreqDec | Row 2 |
| +RD+FD | ✓ | ✓ | ✗ | ✗ | `ablations/ablation_no_distill.py` + `ablation_no_diffprior.py` | Row 3 |
| +RD+FD+DP | ✓ | ✓ | ✓ | ✗ | `ablations/ablation_no_distill.py` | Row 4 |
| **Full** | ✓ | ✓ | ✓ | ✓ | `dark_dino_r50_exdark_diffprior.py` | Row 5 |

**运行命令**:

```bash
# --- Row 1: Baseline (标准 DINO R50) ---
# 使用 mmdet 官方 DINO config，在 ExDark 上训练
python tools/train.py configs/dino/dino_4scale_r50_1x_coco.py \
    --work-dir work_dirs/abl_comp/row1_baseline_dino \
    --cfg-options \
        data_root='data/exdark/' \
        model.bbox_head.num_classes=12 \
        train_dataloader.dataset.data_root='data/exdark/' \
        train_dataloader.dataset.ann_file='annotations/exdark_train.json' \
        train_dataloader.dataset.metainfo.classes="['Bicycle','Boat','Bottle','Bus','Car','Cat','Chair','Cup','Dog','Motorbike','People','Table']" \
        val_dataloader.dataset.data_root='data/exdark/' \
        val_dataloader.dataset.ann_file='annotations/exdark_val.json' \
        val_evaluator.ann_file='data/exdark/annotations/exdark_val.json' \
    --gpu-ids 0 1

# --- Row 2: Only RD-Backbone (无 FreqDec, 无 DiffPrior, 无 Distill) ---
# 使用 ablation_no_rd 的反义: 有 RD 但其他都没有
# 即 dark_dino_r50_exdark.py + 移除 FreqDec neck → ChannelMapper
python tools/train.py \
    configs/dark_dino/ablations/ablation_no_freqdec.py \
    --work-dir work_dirs/abl_comp/row2_rd_only \
    --resume work_dirs/exp_main/stage1/epoch_20.pth \
    --gpu-ids 0 1

# --- Row 3: RD + FreqDec (无 DiffPrior, 无 Distill) ---
python tools/train.py \
    configs/dark_dino/ablations/ablation_no_diffprior.py \
    --work-dir work_dirs/abl_comp/row3_rd_fd \
    --resume work_dirs/exp_main/stage1/epoch_20.pth \
    --gpu-ids 0 1

# --- Row 4: RD + FreqDec + DiffPrior (无 Distill) ---
# 从 row3 的 checkpoint 加载，开启 DiffPrior
python tools/train.py \
    configs/dark_dino/dark_dino_r50_exdark_diffprior.py \
    --work-dir work_dirs/abl_comp/row4_rd_fd_dp \
    --model.distill_loss=None \
    --resume work_dirs/abl_comp/row3_rd_fd/epoch_36.pth \
    --gpu-ids 0 1
    # 注意: 这里需要 override distill_loss=None

# --- Row 5: Full Model (全部组件) ---
# 这就是主实验 Stage 3 的结果
python tools/test.py \
    configs/dark_dino/dark_dino_r50_exdark_diffprior.py \
    work_dirs/exp_main/stage3_full/epoch_12.pth \
    --work-dir work_dirs/abl_comp/row5_full
```

**填入论文**: 将每个实验的最佳 mAP 填入 Table 3 的 AP 列，计算 Δ 差值。

---

### 5.2 Table 4: FreqDec 频段消融 ⚠️ 需新建配置

**对应论文**: Section 4.4, Table 4

**目的**: 验证每个频段的贡献及组合效果

**当前状态**: 代码中 FreqDecNeck 支持通过 `low_ratio`/`high_ratio` 控制频段，但缺少"只保留某频段"的开关。**需要在代码中添加频段掩码开关**，或在配置中通过设置极端阈值模拟：

| 实验 | Low band | Mid band | High band | 模拟方式 |
|------|:--------:|:--------:|:---------:|----------|
| Only Low | ✓ | ✗ | ✗ | 设置 `high_ratio=0.01`（几乎无 mid/high） |
| Only Mid | ✗ | ✓ | ✗ | 设置 `low_ratio=0.99, high_ratio=0.99`（只留中间窄带） |
| Only High | ✗ | ✗ | ✓ | 设置 `low_ratio=0.99`（几乎无 low/mid） |
| Low+Mid | ✓ | ✓ | ✗ | 设置 `high_ratio=0.01` |
| Low+High | ✓ | ✗ | ✓ | 需要代码支持（见下方） |
| Mid+High | ✗ | ✓ | ✓ | 设置 `low_ratio=0.99` |
| All (default) | ✓ | ✓ | ✓ | 默认 `low_ratio=0.25, high_ratio=0.75` |

**建议方案**: 在 `FreqDecoupledNeck.__init__()` 中增加 `active_bands=('low', 'mid', 'high')` 参数，forward 时只处理激活的频段。这需要少量代码改动。

**临时替代方案**（不改代码，仅调参数）:

```bash
# --- Only Low ---
python tools/train.py configs/dark_dino/dark_dino_r50_exdark.py \
    --work-dir work_dirs/abl_band/band_low_only \
    --cfg-options model.neck.low_ratio=0.001 model.neck.high_ratio=0.001 \
    --gpu-ids 0 1

# --- Only Mid (近似: 极窄中间带) ---
python tools/train.py configs/dark_dino/dark_dino_r50_exdark.py \
    --work-dir work_dirs/abl_band/band_mid_only \
    --cfg-options model.neck.low_ratio=0.49 model.neck.high_ratio=0.51 \
    --gpu-ids 0 1

# --- Only High ---
python tools/train.py configs/dark_dino/dark_dino_r50_exdark.py \
    --work-dir work_dirs/abl_band/band_high_only \
    --cfg-options model.neck.low_ratio=0.999 model.neck.high_ratio=0.999 \
    --gpu-ids 0 1

# --- Low + Mid (默认无 High) ---
python tools/train.py configs/dark_dino/dark_dino_r50_exdark.py \
    --work-dir work_dirs/abl_band/band_low_mid \
    --cfg-options model.neck.high_ratio=0.001 \
    --gpu-ids 0 1

# --- Mid + High ---
python tools/train.py configs/dark_dino/dark_dino_r50_exdark.py \
    --work-dir work_dirs/abl_band/band_mid_high \
    --cfg-options model.neck.low_ratio=0.999 \
    --gpu-ids 0 1

# --- All (默认) ---
python tools/train.py configs/dark_dino/dark_dino_r50_exdark.py \
    --work-dir work_dirs/abl_band/band_all_default \
    --gpu-ids 0 1
```

---

### 5.3 Fig. 2: 亮度门控分析（可视化）

**对应论文**: Section 4.4, Fig. 2

**目的**: 展示不同亮度下 gate 权重如何自适应变化

**操作**: 在训练好的完整模型上，按图像亮度分组推理，记录每组的 gate 权重平均值。

```python
# 分析脚本 (保存为 tools/analyze_gate.py)
import torch
import json
from mmengine.config import Config
from mmengine.runner import Runner

def analyze_brightness_gate():
    """分析不同亮度下的 gate 权重分布"""
    # 加载模型
    cfg = Config.fromfile('configs/dark_dino/dark_dino_r50_exdark_diffprior.py')
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint('work_dirs/exp_main/stage3_full/epoch_12.pth')

    model = runner.model.cuda().eval()
    neck = model.neck  # FreqDecoupledNeck

    # 按 brightness 分组收集 gate weights
    brightness_groups = {'dark': [], 'medium': [], 'bright': []}

    with torch.no_grad():
        for batch in dataloader:  # ExDark val set
            img = batch['inputs'].cuda()
            # Forward 到 neck
            features = model.backbone(img)
            raw_feats = neck.channel_mapper(features)

            # 计算 brightness (illumination map mean)
            if hasattr(model.backbone, 'decomposer'):
                _, L = model.backbone.decomposer(img)
                brightness = L.mean(dim=(1,2,3)).item()
            else:
                brightness = img.mean().item()

            # Get gate weights
            _ = neck(raw_feats)  # trigger forward to get gate
            w = neck.gate_weights  # (B, 3) softmax weights

            # 分组
            if brightness < 0.3:
                group = 'dark'
            elif brightness < 0.6:
                group = 'medium'
            else:
                group = 'bright'
            brightness_groups[group].append(w.cpu())

    # 打印各组平均 gate 权重
    for group, weights in brightness_groups.items():
        w_stack = torch.cat(weights, dim=0)
        print(f"{group}: Low={w_stack[:,0].mean():.3f}, "
              f"Mid={w_stack[:,1].mean():.3f}, "
              f"High={w_stack[:,2].mean():.3f}")
```

**预期结果**:
- Dark 组: Low/Mid weight > High weight（暗图更依赖低中频）
- Bright 组: 各 band 权重较均匀（亮图信任高频细节）

---

### 5.4 Table 5: DCT 半径敏感性 ⚠️ 需新建配置

**对应论文**: Section 4.4, Table 5

**目的**: 验证默认半径 `(0.25, 0.75)` 是合理的

```bash
# --- ρ_l=0.20, ρ_h=0.70 ---
python tools/train.py configs/dark_dino/dark_dino_r50_exdark.py \
    --work-dir work_dirs/abl_radius/radius_020_070 \
    --cfg-options model.neck.low_ratio=0.20 model.neck.high_ratio=0.70 \
    --resume work_dirs/exp_main/stage1/epoch_20.pth \
    --gpu-ids 0 1

# --- ρ_l=0.25, ρ_h=0.75 (默认, 主实验已有) ---
# 直接使用 stage3_full 结果

# --- ρ_l=0.30, ρ_h=0.80 ---
python tools/train.py configs/dark_dino/dark_dino_r50_exdark.py \
    --work-dir work_dirs/abl_radius/radius_030_080 \
    --cfg-options model.neck.low_ratio=0.30 model.neck.high_ratio=0.80 \
    --resume work_dirs/exp_main/stage1/epoch_20.pth \
    --gpu-ids 0 1
```

---

### 5.5 Table 6: 蒸馏设计消融 ✅ 配置就绪

**对应论文**: Section 4.4, Table 6

**目的**: Cosine vs Relation vs Both

| 实验 | Cosine | Relation | 配置方式 |
|------|:------:|:--------:|----------|
| Only Cosine | ✓ | ✗ | `cosine_weight=1.0, relation_weight=0.0` |
| Only Relation | ✗ | ✓ | `cosine_weight=0.0, relation_weight=1.0` |
| Both (full) | ✓ | ✓ | 默认 `cosine_weight=1.0, relation_weight=0.5` |

```bash
# --- Only Cosine ---
python tools/train.py configs/dark_dino/dark_dino_r50_exdark_distill.py \
    --work-dir work_dirs/abl_distill/cosine_only \
    --cfg-options \
        model.distill_loss.cosine_weight=1.0 \
        model.distill_loss.relation_weight=0.0 \
    --resume work_dirs/exp_main/stage1/epoch_20.pth \
    --gpu-ids 0 1

# --- Only Relation ---
python tools/train.py configs/dark_dino/dark_dino_r50_exdark_distill.py \
    --work-dir work_dirs/abl_distill/relation_only \
    --cfg-options \
        model.distill_loss.cosine_weight=0.0 \
        model.distill_loss.relation_weight=1.0 \
    --resume work_dirs/exp_main/stage1/epoch_20.pth \
    --gpu-ids 0 1

# --- Both (full, 主实验已有) ---
# stage2_distill 结果即为 both
```

---

### 5.6 Table 7: 训练策略消融 ⚠️ 需自定义配置

**对应论文**: Section 4.4, Table 7

**目的**: 验证三阶段策略优于端到端单阶段

| 策略 | 描述 | 配置方式 |
|------|------|----------|
| End-to-end (1 stage) | 所有一起训练，36ep | `freeze_decomposer=False`, 全部 loss 同时开 |
| Stage 1+2 (no Stage 3) | 有预训练 + 检测，无 DiffPrior | Stage 2 的最终结果 |
| Stage 1+2+3 (**Ours**) | 完整三阶段 | Stage 3 最终结果 |

```bash
# --- End-to-end: 所有组件一起训练 ---
# 基于 dark_dino_r50_exdark.py + 开启全部 loss + 不冻结分解器
python tools/train.py configs/dark_dino/dark_dino_r50_exdark.py \
    --work-dir work_dirs/abl_strategy/endtoend \
    --cfg-options \
        model.freeze_decomposer=False \
        model.distill_loss=@configs/dark_dino/dark_dino_r50_exdark_distill.py.model.distill_loss \
    --gpu-ids 0 1

# --- Stage 1+2 (无 Stage 3): 就是 stage2_distill 的最终结果 ---
# 直接引用 work_dirs/exp_main/stage2_distill 的最佳 mAP

# --- Stage 1+2+3: 就是 stage3_full 的最终结果 ---
# 直接引用 work_dirs/exp_main/stage3_full 的最佳 mAP
```

---

### 5.7 Table 8 & 效率分析 ✅ 自动统计

**对应论文**: Section 4.6, Table 8

**操作**:

```bash
# 统计参数量和 FLOPs
python tools/analysis_tools/get_flops.py \
    configs/dark_dino/dark_dino_r50_exdark_diffprior.py \
    --work_dirs/exp_main/stage3_full/epoch_12.pth

# FPS 测试
python tools/analysis_tools/benchmark.py \
    configs/dark_dino/dark_dino_r50_exdark_diffprior.py \
    --work_dirs/exp_main/stage3_full/epoch_12.pth \
    --repeat 200
```

**需要对比的变体**:

| 变体 | 配置 |
|------|------|
| DINO baseline (R50) | 官方 DINO config |
| Dark-DINO R50 w/o DiffPrior | `dark_dino_r50_exdark.py` (enable=False) |
| Dark-DINO R50 full | `dark_dino_r50_exdark_diffprior.py` |
| DINO Swin-L | 官方 DINO Swin-L config |
| Dark-DINO Swin-L full | `dark_dino_swin-l_exdark.py` + DiffPrior |

---

## 六、可视化实验（Fig. 3-6）

### 6.1 Fig. 3: Retinex 分解可视化

```python
# 保存为 tools/visualize_retinex.py
import torch
import cv2
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile('configs/dark_dino/dark_dino_r50_exdark_diffprior.py')
runner = Runner.from_cfg(cfg)
runner.load_checkpoint('work_dirs/exp_main/stage3_full/epoch_12.pth')
model = runner.model.cuda().eval()

# 选择几张有代表性的低光图片
test_images = ['data/exdark/images/val/xxx.jpg', ...]

for img_path in test_images:
    img = cv2.imread(img_path)
    img_tensor = torch.from_numpy(img).permute(2,0,1).float()/255.0
    img_batch = img_tensor.unsqueeze(0).cuda()

    with torch.no_grad():
        R, L = model.backbone.decomposer(img_batch)

    r_np = R[0].cpu().numpy().transpose(1,2,0)  # reflectance
    l_np = L[0].cpu().numpy().transpose(1,2,0)  # illumination
    recon = r_np * l_np  # should ≈ original

    # 保存三联图: Original | Reflectance | Illumination | Reconstruction
    ...
```

### 6.2 Fig. 4: DCT 频谱可视化

类似地，在 FreqDecNeck forward 过程中 hook 出三个频段的特征图，分别可视化。

### 6.3 Fig. 5: t-SNE 特征分布

```python
# 提取 backbone features → t-SNE 降维 → 着色区分 dark/normal
# 需要: (a) 暗图经过 student backbone 的特征
#       (b) 正常光图经过 teacher (DINOv2) 的特征
# 对比: 有/无 distillation 的特征聚类情况
```

### 6.4 Fig. 6: 定性检测结果

```bash
# 推理并保存检测结果可视化
python tools/infer.py \
    configs/dark_dino/dark_dino_r50_exdark_diffprior.py \
    work_dirs/exp_main/stage3_full/epoch_12.pth \
    --show-dir work_dirs/vis_results \
    --show \
    --data/your_test_image.jpg
```

---

## 七、实验执行顺序建议（推荐顺序）

为避免重复训练和 GPU 浪费，**严格按以下顺序执行**：

### Phase 0: 数据集准备 [Day 1]
- [ ] 下载 ExDark，转换为 COCO 格式
- [ ] 下载 LOL，确认 low/high 配对结构
- [ ] 下载 DarkFace，转换标注格式
- [ ] 验证所有数据加载 pipeline 正常工作（`python tools/test_pipeline.py`）

### Phase 1: 基础训练 [Day 2-5]
- [ ] **Step 1.1**: Stage 1 pretrain (LOL, ~4h on 2×A800)
- [ ] **Step 1.2**: DINO baseline on ExDark (Row 1 of Table 3, ~24h)
- [ ] **Step 1.3**: Stage 2 base detection (RD+FreqDec only, ~28h)
- [ ] **Step 1.4**: Stage 2 with distill (~30h)
- [ ] **Step 1.5**: Stage 3 full model with DiffPrior (~10h)
- **产出**: Table 1 主结果 + Table 3 Row 1/3/4/5 + Table 7 Row 2/3

### Phase 2: 消融实验 [Day 6-9]
- [ ] **Step 2.1**: Row 2 (RD-only) — 需单独训练 (~26h)
- [ ] **Step 2.2**: Table 4 (频段消融) × 7 组 (~26h×7≈182h, 可并行多组)
- [ ] **Step 2.3**: Table 5 (DCT半径) × 3 组 (~26h×3≈78h)
- [ ] **Step 2.4**: Table 6 (蒸馏设计) × 3 组 (~30h×3≈90h)
- [ ] **Step 2.5**: Table 7 (训练策略) — End-to-end (~32h)
- **产出**: Table 3 完成 + Table 4/5/6/7 完成

### Phase 3: 评估与可视化 [Day 10-11]
- [ ] **Step 3.1**: DarkFace 泛化评估 (Table 2)
- [ ] **Step 3.2**: FLOPs/FPS/Params 统计 (Table 8)
- [ ] **Step 3.3**: Fig. 3 (Retinex 分解可视化)
- [ ] **Step 3.4**: Fig. 4 (DCT 频谱可视化)
- [ ] **Step 3.5**: Fig. 5 (t-SNE)
- [ ] **Step 3.6**: Fig. 6 (检测结果对比)

### Phase 4: 填写论文 [Day 11-12]
- [ ] 将所有数值填入 `paper.md` 对应表格
- [ ] 更新 Abstract 中的 XX.X 数值
- [ ] 生成论文 PDF（如使用 LaTeX）

---

## 八、快速参考：配置文件 ↔ 论文表格 映射总表

| 论文表格 | 行/列 | 配置文件 | 是否已就绪 | 备注 |
|----------|-------|----------|:----------:|------|
| **Table 1** | 全部 | `dark_dino_r50_exdark.py` + `_diffprior.py` + `swin-l_exdark.py` | ✅ | 需跑完三阶段 |
| **Table 2** | 全部 | 同上，改 `--cfg-options` 指向 DarkFace | ⚠️ 需改 data_root | |
| **Table 3** | Row 1 | 标准 DINO config | ⚠️ 需改 num_classes=12 | |
| **Table 3** | Row 2 | `ablation_no_freqdec.py` + override | ⚠️ 需确认无 FreqDec | |
| **Table 3** | Row 3 | `ablation_no_diffprior.py` | ✅ | |
| **Table 3** | Row 4 | `dark_dino_r50_exdark_diffprior.py` + distill=None | ⚠️ CLI override | |
| **Table 3** | Row 5 | `dark_dino_r50_exdark_diffprior.py` | ✅ | 主实验结果 |
| **Table 4** | 7行 | `dark_dino_r50_exdark.py` + override `low_ratio`/`high_ratio` | ⚠️ 近似模拟 | |
| **Table 5** | 3行 | 同上 | ⚠️ CLI override | |
| **Table 6** | 3行 | `dark_dino_r50_exdark_distill.py` + override weights | ✅ | |
| **Table 7** | 3行 | 自定义 (e2e / s1+s2 / s1+s2+s3) | ⚠️ e2e 需新配置 | |
| **Table 8** | 全部 | 各配置 + `get_flops.py` + `benchmark.py` | ✅ | |
| **Fig. 2** | - | 自定义分析脚本 | ❌ 需编写 | |
| **Fig. 3-6** | - | 自定义可视化脚本 | ❌ 需编写 | |

---

## 九、常见问题排查

### Q1: CUDA OOM
```bash
# 减小 batch_size (R50: bs=1, Swin-L: 必须bs=1 + gradient checkpointing)
# 启用 gradient checkpointing: 在 backbone config 中加 with_cp=True
# 减小 input resolution: RandomChoiceResize 最大 scale 从 800→640
```

### Q2: SD VAE 下载失败
```bash
# 手动设置镜像
export HF_ENDPOINT=https://hf-mirror.com
# 或手动下载后指定本地路径:
# vae_model='/path/to/local/sd-vae-ft-mse'
```

### Q3: DINOv2 加载慢
```bash
# 首次下载约1.2GB，之后会缓存到 ~/.cache/hub/
# 确保网络通畅或预下载
```

### Q4: 蒸馏损失为 NaN
```bash
# 检查: (a) normal-light 图像是否正确传入
#       (b) teacher feature shape 是否匹配 student projection
#       (c) temperature tau 是否过小 (<1.0 会导致 softmax 溢出)
# 默认 tau=4.0 应该安全
```

### Q5: Stage 1 → Stage 2 权重加载失败 Key mismatch
```bash
# Stage 1 只训练了 decomposer + light_encoder
# Stage 2 加载时使用 --resume (strict=False) 或只加载 matching keys
# 当前代码 DarkDINO.load_pretrained_decomposer() 已处理此逻辑
# 如果仍有问题，检查该函数实现
```
