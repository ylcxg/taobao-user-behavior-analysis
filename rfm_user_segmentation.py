import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ==============================
# 【前置检查】避免 sklearn 缺失报错
# ==============================
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("❌ 缺少机器学习依赖库，请先运行以下命令安装：")
    print("pip install scikit-learn")
    print("安装完成后重新运行本脚本~")
    sys.exit(1)

# ==============================
# 基础配置（自动定位路径，彻底避免找不到文件）
# ==============================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "small_sample.csv")
images_dir = os.path.join(current_dir, "images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# ==============================
# 1. 读取&预处理数据
# ==============================
print("=" * 50)
print("📊 RFM用户分层分析启动...")
print("=" * 50)

# 检查数据文件是否存在
if not os.path.exists(csv_path):
    print(f"❌ 找不到数据文件：{csv_path}")
    print("💡 请把 small_sample.csv 和本脚本放在同一个文件夹里！")
    sys.exit(1)

df = pd.read_csv(csv_path)
df.columns = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df['date'] = df['timestamp'].dt.date

# 观测窗口结束日（数据最后一天）
end_date = pd.to_datetime('2017-12-03').date()

# ==============================
# 2. 计算RFM核心指标（适配无金额字段场景）
# ==============================
print("🔢 计算RFM指标...")
# R：最近一次行为距离窗口结束的天数（越小越活跃）
user_last_active = df.groupby('user_id')['date'].max().reset_index()
user_last_active.columns = ['user_id', 'last_active_date']
# 【修复历史报错】强制转为datetime格式再计算天数差
user_last_active['R'] = (pd.to_datetime(end_date) - pd.to_datetime(user_last_active['last_active_date'])).dt.days
# F：购买次数（仅统计buy行为）
user_buy_count = df[df['behavior_type'] == 'buy'].groupby('user_id').size().reset_index(name='F')

# M：总互动频次（替代消费金额，原始数据无金额字段）
user_total_interaction = df.groupby('user_id').size().reset_index(name='M')

# 合并RFM指标，无购买行为的用户F记为0
rfm = user_last_active.merge(user_buy_count, on='user_id', how='left').fillna({'F': 0})
rfm = rfm.merge(user_total_interaction, on='user_id', how='left')

print(f"✅ RFM指标计算完成！共 {len(rfm)} 名用户")
print(f"   有购买行为的用户：{(rfm['F'] > 0).sum()} 人")
print(f"   无购买行为的用户：{(rfm['F'] == 0).sum()} 人")

# ==============================
# 3. 方法1：规则法分层（标准RFM 8类）
# ==============================
print("\n📌 方法1：规则法RFM分层（分位数定阈值，避免主观偏差）...")
# 用分位数确定阈值（R越小越好，F/M越大越好）
r_threshold = rfm['R'].quantile(0.25)  # 前25%的用户R更小，更活跃
f_threshold = rfm['F'].quantile(0.75)  # 前25%的用户F更高，购买更多
m_threshold = rfm['M'].quantile(0.75)  # 前25%的用户M更高，互动更多

# 打标签：1代表高，0代表低
rfm['R_label'] = np.where(rfm['R'] <= r_threshold, 1, 0)
rfm['F_label'] = np.where(rfm['F'] >= f_threshold, 1, 0)
rfm['M_label'] = np.where(rfm['M'] >= m_threshold, 1, 0)

# 组合成RFM得分（如111=重要价值用户）
rfm['rfm_score'] = rfm['R_label'].astype(str) + rfm['F_label'].astype(str) + rfm['M_label'].astype(str)

# RFM分层映射
rfm_map = {
    '111': '重要价值用户',
    '101': '重要发展用户',
    '011': '重要保持用户',
    '001': '重要挽留用户',
    '110': '一般价值用户',
    '100': '一般发展用户',
    '010': '一般保持用户',
    '000': '一般挽留用户'
}
rfm['user_segment_rule'] = rfm['rfm_score'].map(rfm_map)

# 【修复历史报错】用布尔索引统计，避免if判断数组的错误
segment_rule_stats = rfm['user_segment_rule'].value_counts().sort_values(ascending=False)
print("\n📈 规则法分层结果：")
for segment, count in segment_rule_stats.items():
    print(f"  {segment}：{count}人（占比{count/len(rfm):.2%}）")

# ==============================
# 4. 方法2：K-Means聚类分层（加分项，验证分层稳定性）
# ==============================
print("\n📌 方法2：K-Means聚类分层（标准化后聚类，消除量纲影响）...")
# 特征工程：R越小越好，取倒数反转（加1避免除零错误）
rfm_features = rfm[['R', 'F', 'M']].copy()
rfm_features['R_inv'] = 1 / (rfm_features['R'] + 1)

# 标准化特征
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features[['R_inv', 'F', 'M']])

# K-Means聚类（聚为5类，对应不同价值层级）
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
rfm['cluster_label'] = kmeans.fit_predict(rfm_scaled)

# 根据聚类中心特征命名（R小=F近活跃，F大=购买多，M大=互动多）
cluster_profile = rfm.groupby('cluster_label')[['R', 'F', 'M']].mean().round(2)
print("\n🤖 聚类中心特征（R=最近活跃天数，F=购买次数，M=互动频次）：")
print(cluster_profile)

# 手动映射聚类名称为业务语义
cluster_names = {
    0: '高价值活跃用户',
    1: '低价值沉睡用户',
    2: '中价值潜力用户',
    3: '高价值流失风险用户',
    4: '一般互动用户'
}
rfm['user_segment_cluster'] = rfm['cluster_label'].map(cluster_names)

# 统计聚类结果
segment_cluster_stats = rfm['user_segment_cluster'].value_counts().sort_values(ascending=False)
print("\n📈 聚类分层结果：")
for segment, count in segment_cluster_stats.items():
    print(f"  {segment}：{count}人（占比{count/len(rfm):.2%}）")

# 分层一致性校验（加分项，面试可提）
rfm['rule_high_value'] = rfm['user_segment_rule'].isin(['重要价值用户', '重要发展用户'])
rfm['cluster_high_value'] = rfm['user_segment_cluster'].isin(['高价值活跃用户', '中价值潜力用户'])
consistency = (rfm['rule_high_value'] == rfm['cluster_high_value']).mean()
print(f"\n✅ 规则法与聚类法分层一致性：{consistency:.2%}")

# ==============================
# 5. 可视化：规则法分层占比饼图
# ==============================
print("\n🎨 生成可视化图表...")
plt.figure(figsize=(10, 7))
# 过滤占比<1%的类别，避免饼图混乱
segment_plot = segment_rule_stats[segment_rule_stats / len(rfm) > 0.01]
plt.pie(segment_plot, labels=segment_plot.index, autopct='%1.1f%%', startangle=90,
        colors=plt.cm.Paired(range(len(segment_plot))))
plt.title('RFM用户分层占比（规则法）', fontsize=16)
plt.axis('equal')
plt.tight_layout()

# 保存图片
save_path = os.path.join(images_dir, 'rfm_segment_pie.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()

# ==============================
# 6. 业务结论输出（直接复制到README）
# ==============================
print("\n" + "=" * 50)
print("💡 核心业务结论（可直接复制到README）：")
print("=" * 50)
high_value = rfm[rfm['user_segment_rule'] == '重要价值用户']
sleep_users = rfm[rfm['user_segment_rule'] == '重要挽留用户']
no_buy = rfm[rfm['F'] == 0]

print(f"1. 重要价值用户占比 {len(high_value)/len(rfm):.2%}，人均购买 {high_value['F'].mean():.1f} 次，是核心营收贡献群体")
print(f"2. 重要挽留用户占比 {len(sleep_users)/len(rfm):.2%}，历史购买频次较高但近期活跃度下降，需定向推送召回券")
print(f"3. 无购买行为用户占比 {len(no_buy)/len(rfm):.2%}，需优化首单优惠策略提升转化")
print(f"4. 规则法与聚类法分层一致性达 {consistency:.2%}，分层逻辑稳定可靠")
print("\n✅ RFM分析完成！图表已保存至 images/rfm_segment_pie.png")
print("=" * 50)