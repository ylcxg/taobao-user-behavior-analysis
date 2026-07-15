import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 读数据（和你项目1的路径一致）
df = pd.read_csv('small_sample.csv')
# 如果你之前没重命名列，先跑这句（有的话删掉）
df.columns = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']
# 把时间戳转成datetime（如果项目1没转的话补这句）
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
# 提取「日期」列（只要年月日，去掉时分秒，避免同一天多次访问干扰）
df['date'] = df['timestamp'].dt.date

# 2. 计算每个用户的「首次访问日期」（激活日）
user_first_day = df.groupby('user_id')['date'].min().reset_index()
user_first_day.columns = ['user_id', 'first_day']  # 重命名列，方便后续关联

# 3. 提取每个用户的所有访问日期（去重，一天只算一次）
user_visit_days = df.groupby('user_id')['date'].unique().reset_index()
# 把列表展开，方便判断次日是否访问
user_visit_days['visit_days_list'] = user_visit_days['date'].apply(list)

# 4. 计算次日留存：判断用户是否在「首次访问日+1天」有访问
def calc_retention(row):
    first_day = row['first_day']
    next_day = first_day + pd.Timedelta(days=1)  # 首次日的后一天
    # 如果后一天在用户的访问列表里，就是留存
    return 1 if next_day in row['visit_days_list'] else 0

user_visit_days = user_visit_days.merge(user_first_day, on='user_id', how='left')
user_visit_days['is_retain_next_day'] = user_visit_days.apply(calc_retention, axis=1)

# 5. 统计留存率
total_users = len(user_visit_days)  # 总用户数
retained_users = user_visit_days['is_retain_next_day'].sum()  # 留存用户数
retention_rate = retained_users / total_users * 100

# 6. 打印结果（面试直接能讲的数字）
print(f"总用户数：{total_users}")
print(f"次日留存用户数：{retained_users}")
print(f"次日留存率：{retention_rate:.2f}%")

# 7. 保存可视化图（和项目1风格统一，存到images文件夹）
plt.figure(figsize=(6, 4))
plt.bar(['次日留存', '次日流失'], [retention_rate, 100-retention_rate], color=['#32CD32', '#FF6347'])
plt.title('淘宝用户次日留存率')
plt.ylabel('占比(%)')
# 在柱子上标数字
plt.text(0, retention_rate/2, f'{retention_rate:.1f}%', ha='center', color='white', fontsize=12)
plt.text(1, (100-retention_rate)/2, f'{100-retention_rate:.1f}%', ha='center', color='white', fontsize=12)
plt.ylim(0, 100)
plt.savefig('images/retention_rate.png', dpi=150, bbox_inches='tight')
plt.show()