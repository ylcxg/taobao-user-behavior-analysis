import pandas as pd

# 1. 读取小文件
df = pd.read_csv(r'C:\Users\肖威\Desktop\small_sample.csv')

# 2. 打印所有列名，确认真实名称
print("=== 数据表的真实列名如下 ===")
print(df.columns.tolist())
print("\n")

# 3. 查看前5行数据
print("【前5行数据预览】")
print(df.head())
print("\n")

# 4. 查看数据基本信息（列名、非空值数量、数据类型）
print("【数据基本信息】")
print(df.info())
print("\n")

# 5. 查看各列是否有缺失值
print("【缺失值统计】")
print(df.isnull().sum())
print("\n")

# 6. 查看行为类型的数量分布
# 注意：这里的 '1' 是从上面列名结果中看到的实际列名，请根据你的实际情况修改
print("【行为类型分布】")
print(df['1'].value_counts())
print("\n")

# 7. 将时间戳转换为正常日期格式（Unix时间戳转datetime）
# 假设时间戳列是 '1511544070'，请根据你的列名修改
print("【时间范围统计】")
df['timestamp'] = pd.to_datetime(df['1511544070'], unit='s')
print("最早时间：", df['timestamp'].min())
print("最晚时间：", df['timestamp'].max())
