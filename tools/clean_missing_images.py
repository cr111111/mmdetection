"""
用法: 在服务器上运行
    python clean_missing_images.py

自动检测 data/exdark/images/ 中不存在的图片，
并从 annotations/*.json 中移除对应 images 和 annotations 记录。
"""
import json
import os
import sys
from pathlib import Path


def clean_coco_json(ann_file: str, img_dir: str, output_file: str = None):
    """从 COCO JSON 中移除缺失图片及其标注"""
    
    ann_path = Path(ann_file)
    img_path = Path(img_dir)
    
    if not ann_path.exists():
        print(f"[ERROR] Annotation file not found: {ann_file}")
        return False
    
    with open(ann_path) as f:
        data = json.load(f)
    
    # 找出缺失的图片
    missing_img_ids = set()
    missing_files = []
    for img in data['images']:
        if not (img_path / img['file_name']).exists():
            missing_img_ids.add(img['id'])
            missing_files.append(img['file_name'])
    
    if not missing_img_ids:
        print(f"[OK] {ann_file}: All {len(data['images'])} images exist. No cleanup needed.")
        return True
    
    original_imgs = len(data['images'])
    original_anns = len(data['annotations'])
    
    # 移除缺失图片
    data['images'] = [img for img in data['images'] if img['id'] not in missing_img_ids]
    
    # 移除对应标注
    data['annotations'] = [ann for ann in data['annotations'] 
                           if ann['image_id'] not in missing_img_ids]
    
    removed_imgs = original_imgs - len(data['images'])
    removed_anns = original_anns - len(data['annotations'])
    
    output = output_file or str(ann_path)
    with open(output, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[CLEANED] {ann_file}:")
    print(f"  Missing images:   {removed_imgs} (files: {', '.join(missing_files[:5])}{'...' if len(missing_files)>5 else ''})")
    print(f"  Removed annots:   {removed_anns}")
    print(f"  Remaining images: {len(data['images'])}")
    print(f"  Remaining annots: {len(data['annotations'])}")
    print(f"  Saved to:         {output}")
    return True


def main():
    base_dir = Path(__file__).parent.parent / 'data' / 'exdark'
    img_dir = base_dir / 'images'
    ann_dir = base_dir / 'annotations'
    
    if not img_dir.exists():
        print(f"[ERROR] Image directory not found: {img_dir}")
        sys.exit(1)
    
    # 处理所有 JSON 标注文件
    json_files = list(ann_dir.glob('*.json')) if ann_dir.exists() else []
    
    if not json_files:
        print(f"[ERROR] No JSON files found in {ann_dir}")
        sys.exit(1)
    
    print(f"Image dir: {img_dir}")
    print(f"Found {len(json_files)} annotation file(s)\n")
    
    all_ok = True
    for jf in sorted(json_files):
        ok = clean_coco_json(str(jf), str(img_dir))
        all_ok = all_ok and ok
    
    print()
    if all_ok:
        print("[DONE] All files cleaned successfully.")
    else:
        print("[WARN] Some errors occurred. Check output above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
