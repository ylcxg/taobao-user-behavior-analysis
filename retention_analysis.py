import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 核心修复：自动定位文件路径 ---
# 获取当前脚本所在的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取脚本所在的文件夹
current_dir = os.path.dirname(current_file_path)
# 拼接出数据文件的完整路径（绝对路径）
csv_path = os.path.join(current_dir, 'small_sample.csv')

# 设置绘图风格和字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用于正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用于正常显示负号

print("=" * 30)
print("1. 正在读取数据...")

# --- 增加文件存在性检查 ---
if not os.path.exists(csv_path):
    print(f"❌ 致命错误：在当前目录下找不到 small_sample.csv")
    print(f"请确认：'{csv_path}' 这个文件是否存在？")
    print("💡 提示：请把 small_sample.csv 和本脚本放在同一个文件夹里！")
    exit() # 停止运行

# 读取数据
df = pd.read_csv(csv_path)

# 强制标准化列名（解决可能的列名识别异常）
df.columns = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']

# 转换时间戳
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df['date'] = df['timestamp'].dt.date

print(f"✅ 数据读取成功！共 {len(df)} 条记录。")
print("=" * 30)

# --- 2. 计算真正的次日留存率 ---

# 步骤1：找出每个用户首次访问的日期
user_first_day = df.groupby('user_id')['date'].min().reset_index()
user_first_day.columns = ['user_id', 'first_day']

# 步骤2：找出这些用户第二天有没有来
# 构造第二天日期
user_first_day['next_day'] = user_first_day['first_day'] + pd.Timedelta(days=1)

# 把所有用户的访问日期存成集合，方便快速查找
user_visit_dict = df.groupby('user_id')['date'].apply(set).to_dict()

retained_users = []
for index, row in user_first_day.iterrows():
    user_id = row['user_id']
    next_day_date = row['next_day']
    
    # 如果这个用户第二天有访问记录
    if user_id in user_visit_dict and next_day_date in user_visit_dict[user_id]:
        retained_users.append(user_id)

# 步骤3：计算留存率
total_new_users = len(user_first_day)
retained_count = len(retained_users)
retention_rate = (retained_count / total_new_users * 100)

print(f"📊 次日留存分析结果：")
print(f"   首日新增用户数：{total_new_users}")
print(f"   次日回访用户数：{retained_count}")
print(f"   👉 真正的次日留存率：{retention_rate:.2f}%")

# --- 3. 可视化 ---
print("正在生成图表...")
# 确保 images 文件夹存在
images_dir = os.path.join(current_dir, 'images')
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

plt.figure(figsize=(6, 4))
categories = ['次日留存', '次日流失']
values = [retention_rate, 100 - retention_rate]
colors = ['#32CD32', '#FF6347'] # 绿/红

bars = plt.bar(categories, values, color=colors, width=0.6)

plt.title('淘宝活跃用户次日回访率', fontsize=14)
plt.ylabel('占比(%)', fontsize=12)
plt.ylim(0, 100)

# 在柱子上标数字
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=12, color='white')

plt.tight_layout()

# 保存图片
save_path = os.path.join(images_dir, 'true_retention_rate.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"分析完成！图表已保存至: {save_path}")
print("=" * 30)
