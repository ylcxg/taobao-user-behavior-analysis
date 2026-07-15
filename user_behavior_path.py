import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 【自动定位文件路径，彻底解决找不到文件的问题】
# ==========================================
# 获取当前脚本所在的文件夹路径（Python内置变量，绝对不会出错）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接数据文件的完整路径（不管你把文件夹放哪，都能找到同目录的csv）
csv_path = os.path.join(current_dir, "small_sample.csv")

# 先检查文件是否存在，不存在就友好提示，避免崩溃
if not os.path.exists(csv_path):
    print("❌ 致命错误：找不到 small_sample.csv 文件！")
    print(f"👉 请确认：{csv_path} 这个文件是否存在？")
    print("💡 解决方法：把 small_sample.csv 和本脚本放在同一个文件夹里！")
    exit()  # 停止运行，避免后续报错

# ==========================================
# 1. 读取并预处理数据（自带容错）
# ==========================================
print("📂 正在读取数据...")
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"❌ 读取文件失败：{e}")
    exit()

# 强制标准化列名（解决历史列名识别异常问题）
df.columns = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']
# 转换时间戳为 datetime 格式（秒级时间戳转日期）
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
# 提取日期（方便后续按天统计）
df['date'] = df['timestamp'].dt.date
print(f"✅ 数据读取成功！共 {len(df)} 条记录\n")

# ==========================================
# 2. 核心分析：用户行为路径
# ==========================================
print("="*50)
print("📊 【用户行为路径分析结果】")
print("="*50)

# ------------------------------
# 2.1 各行为用户覆盖率（有多少比例的用户产生过该行为）
# ------------------------------
# 按「用户+行为类型」分组，统计每个用户是否有某类行为
user_behavior = df.groupby(['user_id', 'behavior_type']).size().unstack(fill_value=0)
# 大于0代表用户产生过该行为，求均值就是覆盖率
behavior_coverage = (user_behavior > 0).mean() * 100

print("\n📈 各行为用户覆盖率：")
behavior_map = {'pv':'浏览', 'cart':'加购', 'fav':'收藏', 'buy':'购买'}
for behavior, rate in behavior_coverage.items():
    cn_name = behavior_map.get(behavior, behavior)
    print(f"  {behavior}（{cn_name}）：{rate:.2f}%")

# ------------------------------
# 2.2 典型行为路径TOP5（用户从逛到买的操作顺序）
# ------------------------------
# 按用户+时间排序，得到每个用户的行为序列
df_sorted = df.sort_values(['user_id', 'timestamp'])
# 提取每个用户的行为序列（去重，按时间顺序）
user_path = df_sorted.groupby('user_id')['behavior_type'].apply(lambda x: '→'.join(x.unique()))
top_paths = user_path.value_counts().head(5)

print("\n🛒 典型行为路径TOP5：")
for path, count in top_paths.items():
    # 转成中文，方便阅读
    cn_path = '→'.join([behavior_map.get(p, p) for p in path.split('→')])
    print(f"  {cn_path}：{count}人（占比{count/len(user_path):.2f}%）")

# ------------------------------
# 2.3 购买用户前置行为分析（买之前都干了啥）
# ------------------------------
buy_users = user_behavior[user_behavior['buy'] > 0].index  # 有购买行为的用户
if len(buy_users) > 0:
    pre_buy = df[df['user_id'].isin(buy_users) & (df['behavior_type'] != 'buy')]
    pre_buy_stats = pre_buy.groupby('behavior_type').size() / len(buy_users) * 100
    print("\n💰 购买用户前置行为占比：")
    for behavior, rate in pre_buy_stats.items():
        cn_name = behavior_map.get(behavior, behavior)
        print(f"  {behavior}（{cn_name}）：{rate:.2f}%")

# ==========================================
# 3. 可视化：行为覆盖率柱状图
# ==========================================
print("\n🎨 正在生成可视化图表...")
# 自动创建images文件夹，避免保存失败
images_dir = os.path.join(current_dir, "images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

plt.figure(figsize=(8, 5))
colors = ['#87CEEB', '#FFA500', '#FF69B4', '#32CD32']  # 蓝/橙/粉/绿，区分不同行为
labels = ['浏览(pv)', '加购(cart)', '收藏(fav)', '购买(buy)']

bars = plt.bar(labels, behavior_coverage.values, color=colors, width=0.6)
plt.title('淘宝用户各行为覆盖率', fontsize=14)
plt.ylabel('用户占比(%)', fontsize=12)
plt.ylim(0, max(behavior_coverage.values) * 1.1)

# 在柱子上标注具体百分比，更直观
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
# 保存图片到images文件夹
save_path = os.path.join(images_dir, "behavior_coverage.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ 分析完成！图表已保存至：{save_path}")
print("="*50)