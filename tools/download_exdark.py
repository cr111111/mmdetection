"""
ExDark 数据集格式转换脚本
==========================
将已解压的 ExDark 数据集转换为 mmdet/COCO 格式

数据来源: https://researchdata.um.edu.my/dataset.xhtml?persistentId=doi:10.22452/RD/JUSQEK

数据结构 (解压后):
  _temp_downloads/
  ├── bicycle/Bicycle/          # 图像
  ├── boat/Boat/
  ├── ... (12 个类别目录)
  ├── people/People/
  ├── table/Table/
  └── groundtruth/ExDark_Annno/ # 标注
      ├── Bicycle/              # 2015_00001.png.txt
      ├── Boat/
      └── ... (12 个类别目录)

用法:
    python tools/download_exdark.py [--data-root DATA_ROOT] [--convert-only]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict


# ============================================================
# 配置
# ============================================================

# 类别名称映射 (ExDark 原始名称 -> COCO id)
CATEGORIES = [
    {"id": 1, "name": "Bicycle"},
    {"id": 2, "name": "Boat"},
    {"id": 3, "name": "Bottle"},
    {"id": 4, "name": "Bus"},
    {"id": 5, "name": "Car"},
    {"id": 6, "name": "Cat"},
    {"id": 7, "name": "Chair"},
    {"id": 8, "name": "Cup"},
    {"id": 9, "name": "Dog"},
    {"id": 10, "name": "Motorbike"},
    {"id": 11, "name": "People"},
    {"id": 12, "name": "Table"},
]

NAME_TO_ID = {cat["name"]: cat["id"] for cat in CATEGORIES}
ID_TO_CAT = {cat["id"]: cat for cat in CATEGORIES}

# 图像后缀（统一转小写匹配）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "exdark")


# ============================================================
# 解析器：ExDark bbGt v3 标注文件
# ============================================================

def parse_exdark_annotation(txt_path: Path) -> list:
    """
    解析 ExDark 原生标注文件 (bbGt version=3 格式).

    文件内容示例:
        % bbGt version=3
        Bicycle 204 28 271 193 0 0 0 0 0 0 0
        Bicycle 109 142 191 211 0 0 0 0 0 0 0

    每行格式: class_name xmin ymin width height 0 0 0 0 0 0 0
    bbox 已经是 [x, y, w, h] 格式 (与 COCO 一致)
    """
    objects = []
    if not txt_path.exists():
        return objects

    try:
        lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
    except UnicodeDecodeError:
        try:
            lines = txt_path.read_text(encoding="latin-1").strip().splitlines()
        except Exception:
            return objects

    for line in lines:
        line = line.strip()
        # 跳过空行和注释行
        if not line or line.startswith("%"):
            continue

        parts = line.split()
        if len(parts) < 6:
            continue

        name = parts[0]
        if name not in NAME_TO_ID:
            print(f"      [WARN] Unknown class '{name}' in {txt_path.name}")
            continue

        try:
            xmin = float(parts[1])
            ymin = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue

        if w <= 0 or h <= 0:
            continue

        objects.append({
            "category_id": NAME_TO_ID[name],
            "bbox": [xmin, ymin, w, h],
            "area": w * h,
            "iscrowd": 0,
        })

    return objects


# ============================================================
# 数据收集
# ============================================================

def discover_data(data_root: str):
    """
    扫描已解压的 ExDark 目录结构.

    返回:
        image_map: {image_stem_lower: (image_path, image_size)}
                   其中 image_stem 是 '2015_00001' 这样的编号
        annotation_map: {ann_stem: annotation_objects_list}

    图像位置: _temp_downloads/{lower_cat}/{TitleCaseCat}/{stem}.{ext}
    标注位置: _temp_downloads/groundtruth/ExDark_Annno/{Cat}/{stem}.{ext}.txt
    """
    data_path = Path(data_root)
    temp_dir = data_path / "_temp_downloads"

    if not temp_dir.exists():
        print(f"      [ERROR] Data dir not found: {temp_dir}")
        print(f"      请将解压后的 ExDark 数据放到该路径下")
        return None, None

    gt_dir = temp_dir / "groundtruth" / "ExDark_Annno"
    if not gt_dir.exists():
        print(f"      [WARN] Annotation dir not found: {gt_dir}")
        return None, None

    # ---- 收集所有图像 ----
    image_map = {}   # stem_no_ext -> (full_path, size_hint)
    category_dirs = {
        "bicycle": "Bicycle",
        "boat": "Boat",
        "bottle": "Bottle",
        "bus": "Bus",
        "car": "Car",
        "cat": "Cat",
        "chair": "Chair",
        "cup": "Cup",
        "dog": "Dog",
        "motorbike": "Motorbike",
        "people": "People",
        "table": "Table",
    }

    img_count = 0
    skipped_hidden = 0

    for lower_name, title_name in sorted(category_dirs.items()):
        cat_img_dir = temp_dir / lower_name / title_name
        if not cat_img_dir.exists():
            # 尝试直接在 lower_name 目录下找
            cat_img_dir = temp_dir / lower_name
            if not cat_img_dir.exists():
                print(f"      [WARN] Image dir not found: {lower_name}/{title_name}")
                continue

        for ext in IMAGE_EXTENSIONS:
            for f in cat_img_dir.glob(f"*{ext}"):
                # 跳过 macOS 隐藏文件和资源 fork
                if f.name.startswith("._") or f.name.startswith("."):
                    skipped_hidden += 1
                    continue

                stem = f.stem  # e.g., "2015_00001"
                if stem in image_map:
                    # 同一 stem 的不同后缀，优先保留已存在的
                    continue

                image_map[stem] = f
                img_count += 1

    print(f"      发现图像: {img_count} 张 (跳过隐藏文件: {skipped_hidden})")

    # ---- 收集所有标注 ----
    annotation_map = {}
    ann_count = 0
    missing_ann = 0

    for cat_name in NAME_TO_ID.keys():
        cat_ann_dir = gt_dir / cat_name
        if not cat_ann_dir.exists():
            continue

        for ann_file in cat_ann_dir.glob("*.txt"):
            if ann_file.name.startswith("._") or ann_file.name.startswith("."):
                continue

            # 标注文件名: "2015_00001.png.txt" -> stem = "2015_00001"
            ann_stem = Path(ann_file.stem).stem  # 去掉 .txt 后再去掉 .png 等

            objects = parse_exdark_annotation(ann_file)
            annotation_map[ann_stem] = objects
            ann_count += 1
            if len(objects) == 0:
                missing_ann += 1

    print(f"      发现标注: {ann_count} 个文件 (空标注: {missing_ann})")
    print(f"      有效配对: {len(set(image_map.keys()) & set(annotation_map.keys()))}")

    return image_map, annotation_map


# ============================================================
# 转换: 构建 COCO JSON + 整理图像目录
# ============================================================

def convert_to_coco(image_map, annotation_map, data_root: str):
    """将收集到的数据转换为 COCO 格式并输出."""
    import random
    from PIL import Image

    data_path = Path(data_root)
    out_img_dir = data_path / "images"
    out_ann_dir = data_path / "annotations"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    # 匹配图像-标注对 (取交集)
    common_stems = sorted(set(image_map.keys()) & set(annotation_map.keys()))
    n_total = len(common_stems)

    if n_total == 0:
        print("      ❌ 无有效图像-标注配对!")
        return False

    print(f"\n      转换 {n_total} 个有效样本...")

    # 划分 train/val (~9:1, 固定种子)
    n_val = max(int(n_total * 0.1), 1)
    random.seed(42)
    indices = list(range(n_total))
    random.shuffle(indices)
    val_indices = set(indices[:n_val])

    # 构建 COCO 结构
    coco_train = {"images": [], "annotations": [], "categories": CATEGORIES}
    coco_val = {"images": [], "annotations": [], "categories": CATEGORIES}

    image_id = 0
    ann_id = 0
    cat_counts = defaultdict(int)
    copy_errors = 0
    empty_ann_imgs = 0

    for idx, stem in enumerate(common_stems):
        img_path = image_map[stem]
        objects = annotation_map[stem]

        # 统一输出文件名 (用原始后缀)
        src_suffix = img_path.suffix.lower()
        dst_name = f"{stem}{src_suffix}"
        dst_path = out_img_dir / dst_name

        # 复制图像到统一目录
        if not dst_path.exists():
            try:
                import shutil
                shutil.copy2(str(img_path), str(dst_path))
            except Exception as e:
                copy_errors += 1
                continue

        # 获取图像尺寸
        try:
            with Image.open(dst_path) as im:
                img_w, img_h = im.size
        except Exception:
            img_w, img_h = 0, 0

        image_info = {
            "id": image_id,
            "file_name": dst_name,
            "width": img_w,
            "height": img_h,
        }

        target = coco_val if idx in val_indices else coco_train
        target["images"].append(image_info)

        if len(objects) == 0:
            empty_ann_imgs += 1
        else:
            for obj in objects:
                target["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": obj["category_id"],
                    "bbox": obj["bbox"],
                    "area": obj["area"],
                    "iscrowd": obj["iscrowd"],
                })
                ann_id += 1
                cat_name = ID_TO_CAT[obj["category_id"]]["name"]
                cat_counts[cat_name] += 1

        image_id += 1

        # 进度显示
        if (idx + 1) % 500 == 0 or idx + 1 == n_total:
            pct = (idx + 1) * 100 // n_total
            print(f'      [{ "#" * (pct // 2)}{"." * (50 - pct // 2)}] '
                  f'{pct}% ({idx+1}/{n_total})', end='\r')

    print()

    # ---- 保存 JSON ----
    train_json = out_ann_dir / "exdark_train.json"
    val_json = out_ann_dir / "exdark_val.json"

    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump(coco_train, f, indent=2, ensure_ascii=False)
    with open(val_json, 'w', encoding='utf-8') as f:
        json.dump(coco_val, f, indent=2, ensure_ascii=False)

    # ---- 统计输出 ----
    total_anns = len(coco_train['annotations']) + len(coco_val['annotations'])
    print(f"\n      ╔══════════════════════════════════════╗")
    print(f"      ║       转换完成                        ║")
    print(f"      ╠══════════════════════════════════════╣")
    print(f"      ║  训练集: {len(coco_train['images']):>5} 图, "
          f"{len(coco_train['annotations']):>6} 标注       ║")
    print(f"      ║  验证集: {len(coco_val['images']):>5} 图, "
          f"{len(coco_val['annotations']):>6} 标注         ║")
    print(f"      ║  空标注图: {empty_ann_imgs:>5} 张                  ║")
    print(f"      ║  复制失败: {copy_errors:>5} 张                  ║")
    print(f"      ╠══════════════════════════════════════╣")
    print(f"      ║  各类别统计:                           ║")
    for name in [c["name"] for c in CATEGORIES]:
        cnt = cat_counts.get(name, 0)
        bar = "#" * max(cnt // 20, 1) if cnt > 0 else "-"
        print(f"      ║    {name:<10} {cnt:>5}  {bar}     ")
    print(f"      ╠══════════════════════════════════════╣")
    print(f"      ║  输出目录:                            ║")
    print(f"      ║    {out_img_dir}/")
    n_imgs = len(list(out_img_dir.iterdir()))
    print(f"      ║      images/     ({n_imgs} files)             ║")
    print(f"      ║    {out_ann_dir}/")
    print(f"      ║      exdark_train.json                 ║")
    print(f"      ║      exdark_val.json                   ║")
    print(f"      ╚══════════════════════════════════════╝\n")

    return True


# ============================================================
# 验证
# ============================================================

def verify_dataset(data_root: str):
    """验证生成的 COCO 数据集完整性."""
    data_path = Path(data_root)
    ann_dir = data_path / "annotations"
    img_dir = data_path / "images"

    train_json = ann_dir / "exdark_train.json"
    val_json = ann_dir / "exdark_val.json"

    if not train_json.exists() or not val_json.exists():
        print(f"      [ERROR] Missing annotation files!")
        return False

    with open(train_json, encoding='utf-8') as f:
        train = json.load(f)
    with open(val_json, encoding='utf-8') as f:
        val = json.load(f)

    missing = []
    for img in train['images'] + val['images']:
        fp = img_dir / img['file_name']
        if not fp.exists():
            missing.append(img['file_name'])

    status = "PASS" if len(missing) == 0 else f"WARN: {len(missing)} MISSING"
    print(f"      === 验证结果 ===")
    print(f"      Train: {len(train['images'])} imgs, {len(train['annotations'])} anns")
    print(f"      Val:   {len(val['images'])} imgs, {len(val['annotations'])} anns")
    print(f"      Missing images: {len(missing)}")
    print(f"      Status: {status}")

    if missing:
        print(f"      缺失文件前5个:")
        for m in missing[:5]:
            print(f"        - {m}")

    # mmdet 导入测试
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    try:
        from mmdet.datasets import CocoDataset
        print(f"      mmdet CocoDataset: OK")
    except ImportError as e:
        print(f"      mmdet CocoDataset: SKIP ({e})")

    return len(missing) == 0


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ExDark 数据集格式转换工具 (已解压版本)")
    parser.add_argument(
        "--data-root", type=str, default=DEFAULT_DATA_ROOT,
        help=f"ExDark 数据根目录 (默认: {DEFAULT_DATA_ROOT})")
    parser.add_argument(
        "--skip-copy", action="store_true",
        help="仅生成标注 JSON，不复制图像文件 (用于调试)")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    data_path = Path(data_root)

    print("=" * 60)
    print("  Dark-DINO: ExDark Dataset Converter")
    print("=" * 60)
    print(f"  Data Root: {data_root}\n")

    # Step 1: 扫描数据
    print("[1/2] Scanning dataset structure...")
    image_map, annotation_map = discover_data(data_root)

    if image_map is None or annotation_map is None:
        print("\n  Failed. Check that _temp_downloads/ exists with correct structure.")
        print("  Expected:")
        print("    _temp_downloads/")
        print("    ├── bicycle/Bicycle/     (images)")
        print("    ├── boat/Boat/")
        print("    ├── ...")
        print("    └── groundtruth/ExDark_Annno/")
        print("        ├── Bicycle/          (annotations *.txt)")
        print("        ├── Boat/")
        print("        └── ...")
        return

    # Step 2: 转换
    print("\n[2/2] Converting to COCO format...")
    ok = convert_to_coco(image_map, annotation_map, data_root)

    if not ok:
        print("\n  Conversion failed.")
        return

    # Step 3: 验证
    verify_dataset(data_root)

    print("\n" + "=" * 60)
    print("  Done!")
    print(f"  Update your config: model.data_preprocessor.data_root = '{data_root}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
