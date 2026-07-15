# ExDark 低对比度目标检测可比实验

本目录提供一套可复现的三组对照实验，用于区分：原始低照度检测能力、传统图像增强带来的收益，以及检测感知低照度建模的收益。

## 实验设计

| 实验 | 配置 | 输入与方法 | 目的 |
| --- | --- | --- | --- |
| A：原始基线 | `dino_r50_exdark_baseline.py` | 原始 ExDark 图像 + DINO-R50 | 建立不做低照度处理的检测基线。 |
| B：增强前处理 | `dino_r50_exdark_clahe.py` | CLAHE 增强 + 与 A 完全相同的 DINO-R50 | 判断简单对比度增强是否真的提升检测。 |
| C：检测感知方法 | `dark_dino_r50_exdark.py` | Dark-DINO（Retinex 分解 + 频率解耦颈部） | 衡量检测器内部的低照度表征学习收益。 |

除模型/增强策略外，三组实验共用：ExDark 的 COCO 标注划分、类别定义、随机种子（3407）、训练轮数（36）、批量大小（每卡 2）、数据增强、优化器和学习率调度。

## 数据准备

预期目录如下：

```text
data/exdark/
├── images/
└── annotations/
    ├── exdark_train.json
    └── exdark_val.json
```

数据集使用 12 类：`Bicycle`、`Boat`、`Bottle`、`Bus`、`Car`、`Cat`、`Chair`、`Cup`、`Dog`、`Motorbike`、`People`、`Table`。

如数据还未转换为 COCO 格式，可先检查并运行 `tools/download_exdark.py` 的相关说明。不要改变三组实验的训练/验证划分。

## Linux 双卡一键运行

已为单机两张 A800 提供顺序执行三组实验的一键脚本：`tools/run_low_contrast_experiments.sh`。三组实验各自独占两张 GPU，以避免资源竞争；默认启用 AMP、训练 36 epoch、在每轮验证中保留 `coco/bbox_mAP` 最优 checkpoint，并在训练后自动分布式测试。

```bash
cd /path/to/mmdetection
chmod +x tools/run_low_contrast_experiments.sh
CUDA_VISIBLE_DEVICES=0,1 ./tools/run_low_contrast_experiments.sh
```

运行前请激活包含 `torch`、`mmengine`、`mmdet` 的 MMDetection 环境，并确认 `data/exdark/` 下已有图像和 COCO 标注。结果默认写入 `work_dirs/low_contrast/`：每个实验目录下有 checkpoint 与 `eval/`，完整日志在 `work_dirs/low_contrast/logs/`。

可选环境变量：

```bash
# 中断后续跑；关闭自动评测；或改变结果目录。
RESUME=1 CUDA_VISIBLE_DEVICES=0,1 ./tools/run_low_contrast_experiments.sh
RUN_TEST=0 CUDA_VISIBLE_DEVICES=0,1 ./tools/run_low_contrast_experiments.sh
WORK_ROOT=/mnt/exp/exdark_low_contrast CUDA_VISIBLE_DEVICES=0,1 ./tools/run_low_contrast_experiments.sh
```

## 训练

在仓库根目录执行。单卡示例：

```powershell
python tools/train.py configs/low_contrast_detection/dino_r50_exdark_baseline.py --work-dir work_dirs/low_contrast/a_raw_dino --amp
python tools/train.py configs/low_contrast_detection/dino_r50_exdark_clahe.py --work-dir work_dirs/low_contrast/b_clahe_dino --amp
python tools/train.py configs/low_contrast_detection/dark_dino_r50_exdark.py --work-dir work_dirs/low_contrast/c_dark_dino --amp
```

多卡训练时，保持三组实验的 GPU 数一致。例如：

```powershell
$env:NNODES=1
$env:NPROC_PER_NODE=2
.	ools\dist_train.sh configs/low_contrast_detection/dino_r50_exdark_baseline.py 2 --work-dir work_dirs/low_contrast/a_raw_dino
```

如果 Windows 环境不具备 `dist_train.sh`，请使用 PyTorch 分布式启动方式，或先以单卡完成对照。

## 测试与结果记录

每组选择验证集 `coco/bbox_mAP` 最优的 checkpoint 进行测试：

```powershell
python tools/test.py configs/low_contrast_detection/dino_r50_exdark_baseline.py work_dirs/low_contrast/a_raw_dino/best_coco_bbox_mAP_epoch_*.pth
python tools/test.py configs/low_contrast_detection/dino_r50_exdark_clahe.py work_dirs/low_contrast/b_clahe_dino/best_coco_bbox_mAP_epoch_*.pth
python tools/test.py configs/low_contrast_detection/dark_dino_r50_exdark.py work_dirs/low_contrast/c_dark_dino/best_coco_bbox_mAP_epoch_*.pth
```

建议汇总下列指标。COCO 评估会直接输出 `bbox_mAP`、`bbox_mAP_50`、`bbox_mAP_75`、`bbox_mAP_s/m/l`。

| 实验 | bbox_mAP | AP50 | AP75 | APsmall | APmedium | APlarge | 参数量 | 单图延迟 | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A：Raw DINO |  |  |  |  |  |  |  |  |  |
| B：CLAHE + DINO |  |  |  |  |  |  |  |  |  |
| C：Dark-DINO |  |  |  |  |  |  |  |  |  |

## 公平性约束

1. 训练、验证划分和类别映射必须一致。
2. A/B 仅允许图像增强不同；禁止改变检测器、训练时长、尺度、随机种子或超参数。
3. C 与 A/B 共用训练日程；若显存不足而改变批量大小，必须按全局批量大小线性调整学习率，并在结果表中说明。
4. CLAHE 是确定性增强，训练和推理均启用，避免训练/部署域不一致。
5. 除总 AP 外，重点比较 `APsmall`。低对比度弱小目标通常最容易漏检。

## 结果解读

- **B 高于 A，C 未明显高于 B**：传统增强已解决主要可见性问题；应检查 Dark-DINO 的训练稳定性和模块消融。
- **C 高于 A/B，特别是 APsmall 提升明显**：支持“检测感知特征建模比前置增强更有效”的结论。
- **B 低于 A**：说明 CLAHE 放大了噪声或改变了纹理统计；这也是低照度增强不应只看视觉质量的典型证据。
- **三组均低**：优先检查标注、类别映射、训练收敛曲线与极暗样本比例，再扩展至多曝光或 RGB-T 数据。

## 实现说明

`CLAHEEnhance` 定义于 `mmdet/datasets/transforms/low_light.py`，对 BGR 图像的 Lab 亮度通道应用 CLAHE。它仅是可复现、无额外模型权重的增强对照，不代表 Zero-DCE、Retinexformer 或 HVI 等学习式增强器。

后续扩展学习式增强时，建议新增独立配置并固定 A 的检测器、数据与训练日程；不要用离线覆盖原图的方式替换 `data/exdark/images/`。
