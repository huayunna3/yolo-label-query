# YOLO Label Query Tool

一个用于查询和分析YOLO数据集标签的Python工具。

## 功能特点

- 将YOLO标注数据导入pandas DataFrame
- 提供丰富的pandas查询指令示例
- 支持JSON格式的标注数据处理
- 简单的交互式界面

## 安装依赖

```bash
pip install pandas
```

## 使用方法

1. 运行脚本：
```bash
python 标签数查询.py
```

2. 脚本会自动加载示例数据到pandas DataFrame

3. 使用pandas查询指令进行分析，例如：
```python
# 查看数据
df.head()

# 统计各类标签数量
df['transcription'].value_counts()

# 筛选包含特定字符的标注
df[df['transcription'].str.contains('⌀')]

# 查看困难样本
df[df['difficult'] == True]
```

## 数据格式

脚本支持以下格式的YOLO标注数据：
```json
[
  {
    "transcription": "文本内容",
    "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "difficult": false
  }
]
```

## 示例查询

脚本中包含了详细的pandas查询指令注释，包括：
- 基本统计查询
- 条件筛选
- 字符串操作
- 分组统计
- 排序和过滤

## 许可证

MIT License