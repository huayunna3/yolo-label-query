# YOLO数据集转换为Ultralytics OBB格式处理方案

## 1. 数据现状分析

### 原始数据结构
- **图片文件**: pic_001.png 到 pic_012.png (共12张)
- **标注文件**: sort-table.txt
- **标注格式**: JSON格式，包含transcription、points、difficult字段
- **坐标类型**: 四点坐标 `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]`

### 数据示例
```json
{
  "transcription": "▽Ra3.2",
  "points": [[187, 525], [398, 525], [398, 626], [187, 626]],
  "difficult": false
}
```

## 2. 目标格式要求

### YOLO26-OBB标准格式
```
dataset_root/
├── train/
│   ├── images/      # 训练图片
│   └── labels/      # 训练标签 (.txt)
├── val/
│   ├── images/      # 验证图片
│   └── labels/      # 验证标签 (.txt)
└── data.yaml         # 数据集配置文件
```

### 标注文件格式 (每行)
```
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

### 坐标归一化公式
```python
normalized_x = pixel_x / image_width
normalized_y = pixel_y / image_height
```

## 3. 处理流程

### 3.1 数据解析与分类
1. **解析sort-table.txt**
   - 读取并解析JSON格式的标注数据
   - 提取图片路径和对应的标注信息

2. **类别映射**
   - 收集所有不同的transcription文本
   - 为每个transcription分配唯一的类别ID (0, 1, 2, ...)
   - 创建类别名称映射表

3. **特殊标签处理**
   - `待识别`: 建议单独处理或标注为"unknown"
   - 过滤difficult=true的样本（可选）

### 3.2 坐标转换
1. **获取图片尺寸**
   - 使用PIL或OpenCV读取每张图片
   - 获取准确的宽度和高度

2. **归一化处理**
   - 将四点坐标除以图片尺寸
   - 结果转换为0-1之间的浮点数
   - 保留6位小数精度

### 3.3 文件结构创建
1. **创建目标目录结构**
   - train/images/ (存放训练图片)
   - train/labels/ (存放训练标签)
   - val/images/ (存放验证图片)
   - val/labels/ (存放验证标签)

2. **复制和处理图片**
   - 将原始图片复制到相应目录
   - 重命名文件确保图片和标签文件名一致

3. **生成标注文件**
   - 为每张图片创建对应的.txt标签文件
   - 文件名与图片同名（去掉扩展名）
   - 格式: `class_id x1 y1 x2 y2 x3 y3 x4 y4`

### 3.4 配置文件生成
创建data.yaml配置文件：
```yaml
path: /path/to/dataset_root
train: train/images
val: val/images

names:
  0: class_name_1
  1: class_name_2
  ...
```

## 4. 关键技术要点

### 4.1 坐标验证
- **四点顺序**: 确保坐标按顺时针或逆时针排列
- **多边形处理**: 如果原始数据超过4个点，取外接矩形顶点
- **越界检查**: 确保归一化坐标在0-1范围内

### 4.2 类别管理
- **自动编号**: 根据transcription出现频率排序
- **特殊字符**: 处理包含特殊符号的文本（如▽, ⌀, ↗）
- **重复标签**: 相同transcription使用相同类别ID

### 4.3 数据集划分
- **训练集/验证集**: 建议按8:2或7:3划分
- **分层采样**: 确保各类别在训练集和验证集中都有样本
- **随机种子**: 设置固定随机种子确保可复现

### 4.4 大图处理
- **图像尺寸**: 您的图纸分辨率很高（如4096×2896）
- **建议切片**: 使用DOTA等工具进行切片后再训练
- **或者缩放**: 按比例缩放到合理尺寸（如1024×1024）

## 5. 实现方案

### 方案A: 完整Python转换脚本
```python
import os
import json
from PIL import Image
import shutil
from collections import defaultdict

def convert_to_yolo_obb(sort_table_path, output_dir, train_ratio=0.8):
    """
    将sort-table.txt转换为YOLO OBB格式
    """
    # 1. 读取原始数据
    with open(sort_table_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 2. 解析数据并收集所有transcription
    data_mapping = {}
    all_transcriptions = set()

    for line in lines:
        parts = line.strip().split('\t', 1)
        if len(parts) < 2:
            continue

        img_info, annotations_json = parts
        annotations = json.loads(annotations_json)

        # 提取图片名称
        img_name = img_info.split('/')[-1]

        # 收集transcription
        for ann in annotations:
            transcription = ann['transcription']
            if transcription != '待识别':  # 过滤待识别
                all_transcriptions.add(transcription)

        data_mapping[img_name] = annotations

    # 3. 创建类别映射
    transcriptions = sorted(list(all_transcriptions))
    class_mapping = {trans: idx for idx, trans in enumerate(transcriptions)}

    # 4. 创建目录结构
    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'labels'), exist_ok=True)

    # 5. 处理每张图片
    import random
    random.seed(42)  # 固定随机种子

    for img_name, annotations in data_mapping.items():
        # 获取图片尺寸
        img_path = os.path.join('/e/yolo数据集', img_name)
        img = Image.open(img_path)
        width, height = img.size

        # 决定训练集或验证集
        if random.random() < train_ratio:
            split = 'train'
        else:
            split = 'val'

        # 复制图片
        dst_img_path = os.path.join(output_dir, split, 'images', img_name)
        shutil.copy(img_path, dst_img_path)

        # 生成标签文件
        label_lines = []
        for ann in annotations:
            transcription = ann['transcription']
            if transcription == '待识别':
                continue  # 跳过待识别标签

            points = ann['points']

            # 归一化坐标
            normalized_points = []
            for x, y in points:
                norm_x = round(x / width, 6)
                norm_y = round(y / height, 6)
                normalized_points.extend([norm_x, norm_y])

            # 格式: class x1 y1 x2 y2 x3 y3 x4 y4
            class_id = class_mapping[transcription]
            label_line = f"{class_id} " + " ".join(map(str, normalized_points))
            label_lines.append(label_line)

        # 保存标签文件
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(output_dir, split, 'labels', label_name)

        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(label_lines))

    # 6. 生成data.yaml
    data_yaml_content = f"""path: {output_dir}
train: train/images
val: val/images

names:
"""

    for idx, trans in enumerate(transcriptions):
        data_yaml_content += f"  {idx}: {trans}\n"

    with open(os.path.join(output_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)

    print(f"转换完成！数据集保存到: {output_dir}")
    print(f"类别数量: {len(transcriptions)}")
    print(f"类别映射: {class_mapping}")

# 使用示例
convert_to_yolo_obb(
    sort_table_path='/e/yolo数据集/sort-table.txt',
    output_dir='/e/yolo数据集/yolo26_obb_dataset',
    train_ratio=0.8
)
```

### 方案B: 逐步处理
如果需要更多控制，可以分步骤处理：

1. **数据清洗阶段**
   ```python
   # 分析标注统计
   from collections import Counter
   transcription_counter = Counter([ann['transcription'] for ann in all_annotations])

   # 生成类别统计报告
   for trans, count in transcription_counter.items():
       print(f"{trans}: {count}次")
   ```

2. **坐标验证阶段**
   ```python
   # 验证坐标有效性
   def validate_points(points, width, height):
       for x, y in points:
           if not (0 <= x <= width and 0 <= y <= height):
               return False
       return True
   ```

3. **数据增强阶段**（可选）
   ```python
   # 对小样本类别进行数据增强
   # 旋转、缩放、亮度调整等
   ```

## 6. 质量检查清单

### 6.1 格式检查
- [ ] 每个图片都有对应的标注文件
- [ ] 标注文件名与图片文件名一致（去掉扩展名）
- [ ] 标注文件格式正确（每行9个值）
- [ ] 归一化坐标在0-1范围内
- [ ] 类别ID从0开始连续编号

### 6.2 数据完整性检查
- [ ] 训练集和验证集都不为空
- [ ] 所有类别在训练集中都有样本
- [ ] 无损坏的图片文件
- [ ] 标注点数等于或少于原始点数（去重后）

### 6.3 业务逻辑检查
- [ ] "待识别"标签已正确过滤
- [ ] 困难样本已正确处理
- [ ] 特殊字符类别名称正确处理
- [ ] 多边形坐标顺序正确

## 7. 训练准备

### 7.1 数据集验证
在Ultralytics Platform或本地验证数据集：
```bash
# 安装Ultralytics
pip install ultralytics

# 验证数据集格式
python -c "
from ultralytics.data.utils import check_det_dataset
check_det_dataset('data.yaml')
"
```

### 7.2 训练启动
```python
from ultralytics import YOLO

# 训练YOLO26-OBB模型
model = YOLO('yolo26n-obb.pt')  # 或 yolo26s-obb.pt, yolo26m-obb.pt
model.train(
    data='data.yaml',
    epochs=100,
    batch=16,
    imgsz=1024,  # 根据实际图片尺寸调整
    device='0',  # GPU索引
    patience=50,
    save_period=10
)
```

### 7.3 训练监控
- 使用TensorBoard监控训练过程
- 定期检查验证集mAP指标
- 根据需要调整学习率和数据增强策略

## 8. 常见问题解决

### 问题1: 图片尺寸差异大
**解决方案**: 建议统一缩放或使用切片工具

### 问题2: 类别数量过多
**解决方案**: 合并相似类别或使用文本识别+检测的两阶段方法

### 问题3: "待识别"标签过多
**解决方案**: 作为背景类处理或人工标注后重新处理

### 问题4: 坐标归一化错误
**解决方案**: 使用准确的图片尺寸进行归一化，避免使用固定尺寸

## 9. 上传到Ultralytics Platform

### 压缩数据集
```bash
# 在yolo26_obb_dataset目录下执行
cd /e/yolo数据集/yolo26_obb_dataset
zip -r ../yolo26_obb_dataset.zip .
```

### 上传步骤
1. 访问 https://platform.ultralytics.com/datasets
2. 拖入或选择yolo26_obb_dataset.zip
3. 选择任务类型：OBB (Oriented Bounding Box)
4. 等待平台自动处理和验证

## 10. 后续优化建议

1. **数据质量提升**: 人工复核"待识别"样本
2. **数据集扩充**: 添加更多同类别的工程图纸样本
3. **模型选择**: 根据实际需求选择yolo26n/s/m
4. **超参数调优**: 调整学习率、batch size等参数
5. **集成部署**: 训练完成后导出ONNX格式用于部署