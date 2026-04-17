import os
import json
import ast
from PIL import Image
import shutil
import numpy as np
from collections import Counter
import random
import math

# 定义12个固定类别（按照用户指定顺序）
CLASS_CATEGORIES = [
    "纯数字类",
    "角度类",
    "公差类",
    "直径类",
    "孔类",
    "螺纹类",
    "倒角类",
    "R类",
    "形位公差类",
    "粗糙度类",
    "未识别类",
    "其他类"
]

def classify_transcription(text):
    """按工程含义分类 - 返回类别ID（0-11）"""
    text = str(text).strip()

    # 1. 纯数字类
    if is_pure_number(text):
        return 0  # 纯数字类

    # 2. 角度类
    elif '°' in text:
        return 1  # 角度类

    # 3. 公差类
    elif '±' in text or '^{' in text:
        return 2  # 公差类

    # 4. 直径类
    elif '⌀' in text or 'Ø' in text:
        return 3  # 直径类

    # 5. 孔类
    elif '孔' in text or '销孔' in text:
        return 4  # 孔类

    # 6. 螺纹类
    elif 'M' in text:
        return 5  # 螺纹类

    # 7. 倒角类
    elif text.startswith('C'):
        return 6  # 倒角类

    # 8. R类
    elif text.startswith('R') or 'SR' in text:
        return 7  # R类

    # 9. 形位公差类
    elif any(char in text for char in ['⊥', '∥', '⟂', '↗', '⌭', '⌯', '◎', '⟂']):
        return 8  # 形位公差类

    # 10. 粗糙度类
    elif '▽Ra' in text:
        return 9  # 粗糙度类
    elif '▽' in text:
        return 9  # 粗糙度类

    # 11. 未识别类（作为独立类别）
    elif text == '待识别':
        return 10  # 未识别类

    # 12. 其他类
    else:
        return 11  # 其他类

def is_pure_number(text):
    """判断是否为纯数字（包括小数），修复角度识别问题"""
    try:
        # 移除可能的特殊字符
        clean_text = text.replace('×', '*').replace('X', '*').replace(' ', '')

        # 如果包含°或其他非数字符号，返回False
        if any(char not in '0123456789.+-' for char in clean_text):
            return False

        float(clean_text)
        return True
    except:
        return False

def calculate_rotation_angle(points):
    """
    从4个角点计算旋转角度
    假设第一个点是左上角，计算矩形的主轴角度
    """
    points = np.array(points)

    if len(points) != 4:
        return 0.0  # 不是4个点，默认0度

    # 计算相邻边的向量
    # 假设points顺序为：左上、右上、右下、左下
    # 计算上边的角度
    top_edge = points[1] - points[0]  # [x2-x1, y2-y1]
    top_angle = math.atan2(top_edge[1], top_edge[0]) * 180 / math.pi

    # 计算右边的角度
    right_edge = points[2] - points[1]  # [x3-x2, y3-y2]
    right_angle = math.atan2(right_edge[1], right_edge[0]) * 180 / math.pi

    # 计算下边的角度
    bottom_edge = points[3] - points[2]  # [x4-x3, y4-y3]
    bottom_angle = math.atan2(bottom_edge[1], bottom_edge[0]) * 180 / math.pi

    # 计算左边的角度
    left_edge = points[0] - points[3]  # [x1-x4, y1-y4]
    left_angle = math.atan2(left_edge[1], left_edge[0]) * 180 / math.pi

    # 使用上边角度作为主角度
    rotation = top_angle

    return rotation

def get_rotated_bounding_box(points):
    """
    计算多边形的外接旋转矩形（OBB）
    返回4个角点
    """
    points = np.array(points)

    if len(points) == 4:
        # 如果已经是4点，直接返回
        return points.tolist()

    # 使用最小外接矩形方法
    min_area = float('inf')
    best_rect = None

    # 尝试不同的旋转角度
    for angle in np.linspace(0, 90, 91):  # 0到90度，1度精度
        # 旋转点
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        rotated_points = []
        for x, y in points:
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            rotated_points.append([rx, ry])

        rotated_points = np.array(rotated_points)

        # 找轴对齐的边界框
        min_x, min_y = np.min(rotated_points, axis=0)
        max_x, max_y = np.max(rotated_points, axis=0)

        # 计算面积
        area = (max_x - min_x) * (max_y - min_y)

        if area < min_area:
            min_area = area
            # 创建四个角点（旋转后的坐标系）
            rect_corners = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ])

            # 旋转回原始坐标系
            original_corners = []
            for rx, ry in rect_corners:
                ox = rx * cos_a + ry * sin_a
                oy = rx * sin_a - ry * cos_a
                original_corners.append([ox, oy])

            best_rect = np.array(original_corners)

    return best_rect.tolist()

def convert_to_yolo_obb(sort_table_path, output_dir):
    """
    将sort-table.txt转换为YOLO OBB格式
    - 全部数据用于训练（不分训练集/验证集）
    - 12个固定类别，中文类别名称
    - 未识别作为独立类别
    """
    print("=" * 60)
    print("YOLO OBB数据集转换工具")
    print("=" * 60)

    # 1. 读取原始数据
    with open(sort_table_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 2. 解析数据
    data_mapping = {}
    class_stats = Counter()
    polygon_stats = Counter()

    for line in lines:
        parts = line.strip().split('\t', 1)
        if len(parts) < 2:
            continue

        img_info, annotations_json = parts
        try:
            # 先尝试JSON解析
            annotations = json.loads(annotations_json)
        except json.JSONDecodeError:
            # 如果失败，尝试处理Python格式的布尔值
            annotations_json_fixed = annotations_json.replace('false', 'False').replace('true', 'True')
            try:
                annotations = ast.literal_eval(annotations_json_fixed)
            except:
                print(f"警告：无法解析数据: {img_info}")
                continue

        # 提取图片名称
        img_name = img_info.split('/')[-1]
        data_mapping[img_name] = annotations

        # 统计类别和点数
        for ann in annotations:
            transcription = ann.get('transcription', '')
            class_id = classify_transcription(transcription)
            class_name = CLASS_CATEGORIES[class_id]
            class_stats[class_name] += 1

            # 统计多边形点数
            point_count = len(ann.get('points', []))
            polygon_stats[f"{point_count}点"] += 1

    # 3. 创建目录结构（全部用于训练）
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)

    # 4. 处理每张图片
    conversion_stats = {
        'total_images': 0,
        'converted_annotations': 0,
        'failed_annotations': 0,
        'polygon_processed': Counter()
    }

    for img_name, annotations in data_mapping.items():
        # 获取图片尺寸
        base_dir = os.path.dirname(sort_table_path)
        img_path = os.path.join(base_dir, img_name)

        if not os.path.exists(img_path):
            print(f"警告：图片不存在 {img_name}")
            continue

        img = Image.open(img_path)
        width, height = img.size
        conversion_stats['total_images'] += 1

        # 复制图片到训练集
        dst_img_path = os.path.join(train_images_dir, img_name)
        shutil.copy(img_path, dst_img_path)

        # 生成标签文件
        label_lines = []
        for ann in annotations:
            transcription = ann.get('transcription', '')

            # 获取类别ID（0-11）
            class_id = classify_transcription(transcription)
            class_name = CLASS_CATEGORIES[class_id]

            points = ann.get('points', [])
            point_count = len(points)

            # 处理多边形（非4点需要计算外接矩形）
            if point_count != 4:
                try:
                    points = get_rotated_bounding_box(points)
                    conversion_stats['polygon_processed'][f"{point_count}点转4点"] += 1
                except Exception as e:
                    print(f"警告：多边形处理失败 {img_name} - {transcription}: {e}")
                    continue

            # 归一化坐标
            normalized_points = []
            for x, y in points:
                norm_x = round(x / width, 6)
                norm_y = round(y / height, 6)
                normalized_points.extend([norm_x, norm_y])

            # 格式: class x1 y1 x2 y2 x3 y3 x4 y4
            label_line = f"{class_id} " + " ".join(map(str, normalized_points))
            label_lines.append(label_line)
            conversion_stats['converted_annotations'] += 1

        # 保存标签文件
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_name)

        if label_lines:
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(label_lines))
        else:
            # 如果没有有效的标注，创建空文件
            open(label_path, 'w', encoding='utf-8').close()

    # 5. 生成data.yaml（12个中文类别）
    data_yaml_content = f"""path: {output_dir}
train: train/images
val: train/images

names:
"""

    for class_id, class_name in enumerate(CLASS_CATEGORIES):
        data_yaml_content += f"  {class_id}: {class_name}\n"

    with open(os.path.join(output_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)

    # 6. 输出统计信息
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"数据集保存到: {output_dir}")
    print(f"处理图片数: {conversion_stats['total_images']}")
    print(f"转换标注数: {conversion_stats['converted_annotations']}")
    print(f"失败标注数: {conversion_stats['failed_annotations']}")

    print("\n多边形处理统计:")
    for poly_type, count in conversion_stats['polygon_processed'].items():
        print(f"  {poly_type}: {count}个")

    print("\n类别统计:")
    for class_name, count in class_stats.most_common():
        print(f"  {class_name}: {count}个")

    print("\n类别映射（共12个，按用户指定顺序）:")
    for class_id, class_name in enumerate(CLASS_CATEGORIES):
        print(f"  {class_id}: {class_name}")

    # 7. 生成统计报告
    report_path = os.path.join(output_dir, 'conversion_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("YOLO OBB数据集转换报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"转换时间: {os.path.basename(output_dir)}\n")
        f.write(f"原始图片数: {conversion_stats['total_images']}\n")
        f.write(f"转换标注数: {conversion_stats['converted_annotations']}\n")
        f.write(f"失败标注数: {conversion_stats['failed_annotations']}\n\n")
        f.write("多边形处理统计:\n")
        for poly_type, count in conversion_stats['polygon_processed'].items():
            f.write(f"  {poly_type}: {count}\n")

        f.write("\n类别统计:\n")
        for class_name, count in class_stats.items():
            f.write(f"  {class_name}: {count}\n")

        f.write("\n类别映射（共12个）:\n")
        for class_id, class_name in enumerate(CLASS_CATEGORIES):
            f.write(f"  {class_id}: {class_name}\n")

    print(f"\n详细报告已保存到: {report_path}")
    print("\n下一步:")
    print("1. 检查生成的标注文件格式是否正确")
    print("2. 在Ultralytics Platform上测试上传")
    print("3. 开始模型训练")

if __name__ == "__main__":
    # 使用示例
    convert_to_yolo_obb(
        sort_table_path=r'E:\yolo数据集\sort-table.txt',
        output_dir=r'E:\yolo数据集\yolo26_obb_dataset'
    )