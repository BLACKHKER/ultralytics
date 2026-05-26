"""
LabelMe JSON → YOLOv8-Pose TXT 转换脚本

用法:
    python labelme2yolopose.py --input ./labelme_json --output ./labels

LabelMe 标注规范:
    1. person: 用 rectangle 画 bbox, label 填 "person"
       - 17个关键点用 point 标注，label 填关键点名称（见 KEYPOINT_NAMES）
       - 同一个人的 bbox 和关键点必须设置相同的 group_id
    2. 其他类别(如 agv, glasses): 用 rectangle 画 bbox, 填对应 label（无需标关键点）
       - 需要在 CLASS_MAP 中添加对应类别
"""

import json
import argparse
from pathlib import Path

# 类别映射
# 类别映射 —— 根据实际需求修改
# 示例: person + glasses 测试
CLASS_MAP = {
    'person': 0,
    'glasses': 1,
}

# COCO 17 关键点名称（LabelMe 标注时用这些名称）
KEYPOINT_NAMES = [
    'nose',
    'left_eye',    'right_eye',
    'left_ear',    'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow',  'right_elbow',
    'left_wrist',  'right_wrist',
    'left_hip',    'right_hip',
    'left_knee',   'right_knee',
    'left_ankle',  'right_ankle',
]


def convert_one(json_path, output_dir):
    """转换单个 LabelMe JSON 文件为 YOLOv8-Pose TXT"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w = data['imageWidth']
    img_h = data['imageHeight']
    shapes = data['shapes']

    # ---------- 按 group_id 分组 ----------
    # group_id 相同的 rectangle + points 属于同一个目标
    groups = {}     # {group_id: {'bbox': shape, 'keypoints': {name: shape}}}
    ungrouped = []  # 没有 group_id 的 rectangle（通常是 AGV）

    for s in shapes:
        gid = s.get('group_id')
        label = s['label'].strip().lower()

        if s['shape_type'] == 'rectangle':
            if gid is not None:
                groups.setdefault(gid, {'bbox': None, 'keypoints': {}})
                groups[gid]['bbox'] = s
            else:
                ungrouped.append(s)

        elif s['shape_type'] == 'point':
            if gid is not None:
                groups.setdefault(gid, {'bbox': None, 'keypoints': {}})
                groups[gid]['keypoints'][label] = s

    # ---------- 生成标注行 ----------
    lines = []

    # 处理有 group_id 的目标（person）
    for gid, group in groups.items():
        bbox_shape = group['bbox']
        if bbox_shape is None:
            print(f"  警告: group_id={gid} 有关键点但没有 bbox，跳过")
            continue

        label = bbox_shape['label'].strip().lower()
        cls_id = CLASS_MAP.get(label)
        if cls_id is None:
            print(f"  警告: 未知类别 '{label}'，跳过")
            continue

        line = _make_line(bbox_shape, group['keypoints'], cls_id, img_w, img_h)
        lines.append(line)

    # 处理没有 group_id 的目标（agv 等只有 bbox 的）
    for bbox_shape in ungrouped:
        label = bbox_shape['label'].strip().lower()
        cls_id = CLASS_MAP.get(label)
        if cls_id is None:
            print(f"  警告: 未知类别 '{label}'，跳过")
            continue

        line = _make_line(bbox_shape, {}, cls_id, img_w, img_h)
        lines.append(line)

    # ---------- 写入 TXT ----------
    txt_name = Path(json_path).stem + '.txt'
    out_path = Path(output_dir) / txt_name
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(lines)


def _make_line(bbox_shape, keypoints_dict, cls_id, img_w, img_h):
    """生成一行 YOLOv8-Pose 格式的标注"""
    # bbox: LabelMe rectangle 的两个角点
    pts = bbox_shape['points']
    x1 = min(pts[0][0], pts[1][0])
    y1 = min(pts[0][1], pts[1][1])
    x2 = max(pts[0][0], pts[1][0])
    y2 = max(pts[0][1], pts[1][1])

    # 归一化 (cx, cy, w, h)
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    parts = [f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"]

    # 17 个关键点
    for kp_name in KEYPOINT_NAMES:
        if kp_name in keypoints_dict:
            kp_shape = keypoints_dict[kp_name]
            kx = kp_shape['points'][0][0] / img_w
            ky = kp_shape['points'][0][1] / img_h
            kv = 2  # 可见
            parts.append(f"{kx:.6f} {ky:.6f} {kv}")
        else:
            # 未标注的关键点 → 不可见
            parts.append("0 0 0")

    return ' '.join(parts)


def main():
    parser = argparse.ArgumentParser(description='LabelMe JSON → YOLOv8-Pose TXT')
    parser.add_argument('--input', required=True, help='LabelMe JSON 文件夹路径')
    parser.add_argument('--output', required=True, help='输出 TXT 文件夹路径')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob('*.json'))
    if not json_files:
        print(f"错误: {input_dir} 中没有找到 JSON 文件")
        return

    total_objects = 0
    for jf in json_files:
        count = convert_one(jf, output_dir)
        total_objects += count
        print(f"  {jf.name} → {count} 个目标")

    print(f"\n完成: {len(json_files)} 个文件, 共 {total_objects} 个目标")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()
