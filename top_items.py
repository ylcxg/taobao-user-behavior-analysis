import pandas as pd
import matplotlib.pyplot as plt

# 1.中文字体以及正负号正常显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 2. 读取数据
df = pd.read_csv('small_sample.csv')

# 3. 重命名列，方便后续操作（根据你实际的列名调整）
df.columns = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']

# 4. 筛选出“浏览”行为的数据（pv）
df_pv = df[df['behavior_type'] == 'pv']

df_buy = df[df['behavior_type'] == 'buy']
# 5. 统计每个商品被浏览的次数，取前10名
top10_items = df_pv['item_id'].value_counts().head(10)

# 6. 打印结果
print("=== Top 10 热门商品（被浏览最多） ===")
print(top10_items)
print("购买记录的总条数：", len(df_buy))

# 7. 绘制柱状图
plt.figure(figsize=(10, 6))
top10_items.plot(kind='bar', color='skyblue')
plt.title('Top 10 热门商品（浏览数）')
plt.xlabel('商品ID')
plt.ylabel('浏览次数')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()