# ExDark 数据集下载与准备指南

> **当前状态**: 网络环境限制导致自动下载不可用，请按以下步骤手动完成。

---

## 第一步：下载数据集

### 方法 A：从马来亚大学数据门户下载（推荐，最完整）

1. **浏览器访问**以下地址：
   ```
   https://researchdata.um.edu.my/dataset.xhtml?persistentId=doi:10.22452/RD/JUSQEK
   ```

2. **登录或跳过登录**（部分文件可直接下载）

3. **逐个下载以下 12 个类别压缩包**（约 1.2 GB 总计）：

   | 文件名 | 大小 | 内容 |
   |--------|------|------|
   | bicycle.tar.gz | 117.6 MB | 652 张（自行车） |
   | boat.tar.gz | 91.6 MB | 679 张（船） |
   | bottle.tar.gz | 89.8 MB | 547 张（瓶子） |
   | bus.tar.gz | 116.1 MB | 527 张（公交车） |
   | car.tar.gz | 159.7 MB | 638 张（汽车） |
   | cat.tar.gz | 100.4 MB | 735 张（猫） |
   | chair.tar.gz | 93.1 MB | 648 张（椅子） |
   | cup.tar.gz | 71.6 MB | 519 张（杯子） |
   | dog.tar.gz | 113.1 MB | 801 张（狗） |
   | motorbike.tar.gz | ~90 MB | 503 张（摩托车） |
   | people.tar.gz | ~100 MB | 609 张（人） |
   | table.tar.gz | ~85 MB | 505 张（桌子） |

4. 将所有 `.tar.gz` 文件放到同一目录，例如：
   ```
   d:/my_project/mmdetection/data/exdark/_temp_downloads/
   ```

### 方法 B：从其他镜像下载

- **Kaggle**: 搜索 "ExDark" 或 "Exclusively Dark"
- **百度网盘**: 搜索 "ExDark 数据集"
- **Google Drive**: 搜索 "Exclusively Dark Image Dataset"

### 方法 C：使用迅雷/IDM 等下载工具

如果浏览器下载慢，复制以下直链到下载工具：

```
https://researchdata.um.edu.my/api/access/datafile/<FILE_ID>
```
（FILE_ID 需要从网页上获取）

---

## 第二步：运行转换脚本

下载完成后，运行转换命令：

```bash
cd d:\my_project\mmdetection

# 激活环境
.venv\Scripts\activate

# 运行转换（假设 .tar.gz 已放在 data/exdark/_temp_downloads/）
python tools/download_exdark.py --data-root data/exdark --convert-only
```

### 转换脚本会自动完成：

1. ✅ 解压所有 `.tar.gz` 到临时目录
2. ✅ 收集图像 + XML 标注
3. ✅ VOC → COCO 格式转换
4. ✅ 按 9:1 划分 train/val
5. ✅ 输出目录结构:

```
data/exdark/
├── images/              # 所有低光图像 (7363张)
│   ├── 2015_00000.jpg
│   ├── 2015_00001.jpg
│   └── ...
└── annotations/
    ├── exdark_train.json # 训练集 COCO 标注 (~6627 images)
    └── exdark_val.json   # 验证集 COCO 标注 (~736 images)
```

---

## 第三步：验证安装

```bash
python -c "
import json
with open('data/exdark/annotations/exdark_train.json') as f:
    d = json.load(f)
print(f'Train: {len(d[\"images\"])} images, {len(d[\"annotations\"])} annotations')
print(f'Categories: {[c[\"name\"] for c in d[\"categories\"]]}')
print('OK!')
"
```

预期输出:
```
Train: ~6627 images, ~18000+ annotations
Categories: ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']
OK!
```

---

## 第四步：更新配置文件路径

确认配置文件中的 `data_root` 指向正确位置。编辑以下文件：

```bash
# configs/dark_dino/dark_dino_r50_exdark.py
# configs/dark_dino/dark_dino_swin-l_exdark.py
# configs/dark_dino/stage1_pretrain_rd.py
```

确保每行的 `data_root` 为:
```python
data_root = 'data/exdark/'   # 或绝对路径如 'd:/my_project/mmdetection/data/exdark/'
```

---

## 常见问题

**Q1: 解压后找不到 XML 标注？**
> ExDark 的标注可能在单独的 Annotation 文件中。检查解压后是否有 `Annotation` 或 `Annotations` 目录。
> 如果标注和图像在不同 tar.gz 中，确保全部下载。

**Q2: 类别名称不匹配？**
> ExDark 使用 Pascal VOC 风格的类别名。转换脚本已处理大小写映射。
> 如果出现未知类别警告，检查 XML 中的 `<name>` 字段。

**Q3: 图像数量不对？**
> ExDark 官方: 7363 images total (train=3000, val=1800, test=2563)。
> 我们的划分是随机 9:1（非官方划分），这是论文实验的标准做法。

**Q4: 转换脚本报错？**
> 确保 Python 3.10 环境: `.venv\Scripts\activate`
> 安装依赖: `uv pip install tqdm lxml`

---

## 下载数据后的下一步

数据准备好后，即可开始训练流程（详见 experiment_guide.md）：

```bash
# Stage 1: Retinex 分解预训练
python tools/train.py configs/dark_dino/stage1_pretrain_rd.py --work-dir work_dirs/stage1 --gpu-ids 0 1

# Stage 2: 检测微调
python tools/train.py configs/dark_dino/dark_dino_r50_exdark.py \
    --work-dir work_dirs/stage2 \
    --resume work_dirs/stage1/epoch_20.pth \
    --gpu-ids 0 1

# Stage 3: Diffusion Prior 微调
python tools/train.py configs/dark_dino/dark_dino_r50_exdark_diffprior.py \
    --work-dir work_dirs/stage3_full \
    --resume work_dirs/stage2/epoch_36.pth \
    --gpu-ids 0 1
```
