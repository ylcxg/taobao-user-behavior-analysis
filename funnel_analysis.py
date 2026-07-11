import matplotlib.pyplot as plt

# 1. 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 2. 准备数据（直接从你刚才的运行结果抄下来）
labels = ['浏览量 (PV)', '购买量 (Buy)']
values = [95489, 2101]  # 这里的 95489 是上一轮统计的 pv 总数，2101 是你刚跑出来的 buy 总数

# 3. 设置颜色（浏览用浅蓝，购买用深绿，表示转化）
colors = ['#87CEEB', '#32CD32']

# 4. 绘制柱状图
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=colors, width=0.6)

# 5. 在柱子上显示具体的数字
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=12)

# 6. 添加标题和标签
plt.title('用户行为转化漏斗：浏览 vs 购买')
plt.ylabel('记录条数')
plt.ylim(0, values[0] * 1.1) # 设置Y轴范围，让图更好看

plt.show()