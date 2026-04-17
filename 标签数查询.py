import pandas as pd
import json
import ast

# 您的原始数据
data = [{"transcription": "▽Ra3.2", "points": [[187, 525], [398, 525], [398, 626], [187, 626]], "difficult": False},
        {"transcription": "▽Ra3.2", "points": [[393, 1138], [496, 1138], [496, 1322], [393, 1322]], "difficult": False},
        {"transcription": "▽Ra0.8", "points": [[555, 391], [751, 391], [751, 489], [555, 489]], "difficult": False},
        {"transcription": "▽Ra1.6", "points": [[685, 591], [776, 591], [776, 778], [685, 778]], "difficult": False},
        {"transcription": "14f9", "points": [[817, 1598], [908, 1598], [908, 1640], [817, 1640]], "difficult": False},
        {"transcription": "⌀60h11", "points": [[230, 989], [286, 989], [286, 1116], [230, 1116]], "difficult": False},
        {"transcription": "⌀56", "points": [[306, 1006], [354, 1006], [354, 1098], [306, 1098]], "difficult": False},
        {"transcription": "⌀46", "points": [[668, 1053], [715, 1053], [715, 1140], [668, 1140]], "difficult": False},
        {"transcription": "⌀84", "points": [[1100, 1013], [1146, 1013], [1146, 1096], [1100, 1096]], "difficult": False},
        {"transcription": "⌀88h11", "points": [[1162, 989], [1210, 989], [1210, 1112], [1162, 1112]], "difficult": False},
        {"transcription": "⌀35H7", "points": [[1232, 998], [1285, 998], [1285, 1116], [1232, 1116]], "difficult": False},
        {"transcription": "10D9", "points": [[1620, 1319], [1710, 1319], [1710, 1361], [1620, 1361]], "difficult": False},
        {"transcription": "⌀30H12", "points": [[1608, 1410], [1724, 1410], [1724, 1457], [1608, 1457]], "difficult": False},
        {"transcription": "↗0.02A", "points": [[1019, 1578], [1234, 1578], [1234, 1643], [1019, 1643]], "difficult": False}]

# 导入到pandas DataFrame
df = pd.DataFrame(data)

print("数据已导入到pandas DataFrame")
print(f"数据形状: {df.shape}")
print(f"数据预览:")
print(df.head())
print("\n所有数据:")
print(df)

# ========== pandas查询指令示例（注释） ==========

# 1. 基本统计
# df.info()                    # 查看数据信息
# df.describe()                # 描述性统计
# df.shape                     # 数据形状
# len(df)                      # 行数
# df.columns                   # 列名

# 2. 查看特定列
# df['transcription']          # 查看transcription列
# df[['transcription', 'difficult']]  # 查看多列

# 3. 条件筛选
# df[df['difficult'] == True]            # 筛选difficult为True的行
# df[df['transcription'].str.contains('⌀')]  # 筛选包含直径符号的行
# df[df['transcription'].str.contains('Ra')]  # 筛选包含Ra的行
# df[df['transcription'].str.startswith('▽')] # 筛选以▽开头的行
# df[df['transcription'].str.contains('\d')]  # 筛选包含数字的行

# 4. 统计查询
# df['transcription'].value_counts()      # 统计每个文本出现的次数
# df['difficult'].value_counts()         # 统计difficult值的分布
# df['transcription'].nunique()          # 统计唯一文本数量

# 5. 字符串操作
# df['transcription'].str.len()          # 计算每个文本的长度
# df['transcription'].str.upper()        # 转换为大写
# df['transcription'].str.lower()        # 转换为小写
# df['transcription'].str.replace('⌀', '直径')  # 替换字符串

# 6. 分组统计
# df.groupby('difficult').size()         # 按difficult分组统计
# df.groupby('difficult')['transcription'].count()  # 分组计数

# 7. 排序
# df.sort_values('transcription')        # 按文本排序
# df.sort_values('transcription', ascending=False)  # 降序排序

# 8. 提取坐标信息（points列是列表的列表）
# 获取第一个点的x坐标
# df['points'].apply(lambda pts: pts[0][0] if pts else None)

# 9. 创建新列
# df['text_length'] = df['transcription'].str.len()  # 文本长度列
# df['has_diameter'] = df['transcription'].str.contains('⌀')  # 是否包含直径符号

# 10. 保存数据
# df.to_csv('yolo_data.csv', index=False, encoding='utf-8-sig')  # 保存为CSV
# df.to_excel('yolo_data.xlsx', index=False)  # 保存为Excel

print("\n" + "="*50)
print("pandas查询指令已写在代码注释中")
print("在Python交互环境中可以执行这些指令查询数据")