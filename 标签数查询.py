import pandas as pd
import json
import ast

# 您的原始数据（示例）
data_input=input(" ")
# 方法1：直接使用JSON解析
try:
    # 解析JSON数据
    data_list = json.loads(data_input)
except json.JSONDecodeError:
    # 如果JSON解析失败，尝试使用Python的literal_eval（处理false/true）
    data_input_fixed = data_input.replace('false', 'False').replace('true', 'True')
    data_list = ast.literal_eval(data_input_fixed)

# 转换为pandas DataFrame
df = pd.DataFrame(data_list)

print("数据已成功转换为pandas DataFrame")
print("="*50)
print(f"数据形状: {df.shape} (行数: {len(df)}, 列数: {len(df.columns)})")
print(f"列名: {list(df.columns)}")
print("\n数据预览:")
print(df)

print("\n" + "="*50)
print("pandas查询指令示例:")
print("="*50)
print("""
# 基本操作
df.head()                           # 查看前几行
df.info()                           # 查看数据信息
df.describe()                       # 描述性统计

# 查看特定列
df['transcription']                 # 查看transcription列
df[['transcription', 'difficult']]  # 查看多列

# 条件筛选
df[df['difficult'] == True]         # 筛选困难样本
df[df['transcription'].str.contains('⌀')]  # 筛选包含直径符号的行
df[df['transcription'].str.contains('Ra')]  # 筛选包含Ra的行

# 统计查询
df['transcription'].value_counts()  # 统计每个文本出现的次数
df['difficult'].value_counts()      # 统计difficult值的分布
df['transcription'].nunique()       # 统计唯一文本数量

# 字符串操作
df['text_length'] = df['transcription'].str.len()  # 计算文本长度
df['has_diameter'] = df['transcription'].str.contains('⌀')  # 是否包含直径符号

# 分组统计
df.groupby('difficult').size()      # 按difficult分组统计
df.groupby('difficult')['transcription'].count()  # 分组计数

# 坐标信息提取
# 获取第一个点的x坐标
df['first_x'] = df['points'].apply(lambda pts: pts[0][0] if pts else None)

# 保存数据
# df.to_csv('yolo_data.csv', index=False, encoding='utf-8-sig')
# df.to_excel('yolo_data.xlsx', index=False)
""")
