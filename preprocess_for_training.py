"""
真实数据预处理脚本
从原始CSV文件中提取训练特征，生成 training_data.npz
用于 train_lightgbm.py 进行模型训练
"""
import pandas as pd
import numpy as np
import os

print("=" * 50)
print("真实训练数据预处理")
print("=" * 50)

# ========== 1. 读取原始数据 ==========
print("\n📂 读取原始数据...")

try:
    df_logs = pd.read_csv('b_logs.csv', nrows=500000)
except:
    print("❌ 未找到 b_logs.csv，请确保文件在当前目录下")
    exit(1)

try:
    df_purchase = pd.read_csv('b_purchase.csv')
except:
    print("❌ 未找到 b_purchase.csv，请确保文件在当前目录下")
    exit(1)

print(f"   行为日志: {len(df_logs):,} 条")
print(f"   购买记录: {len(df_purchase):,} 条")

# ========== 2. 特征工程 ==========
print("\n🔧 特征工程...")

# 2.1 行为次数
user_actions = df_logs.groupby('user_id').size().reset_index(name='actions')

# 2.2 浏览商品数
user_browse = df_logs.groupby('user_id')['item_id'].nunique().reset_index(name='browse')

# 2.3 历史购买次数
user_purchases = df_purchase.groupby('user_id').size().reset_index(name='purchases')

# 2.4 合并特征
df = user_actions.merge(user_browse, on='user_id', how='left')\
                  .merge(user_purchases, on='user_id', how='left')

df['purchases'] = df['purchases'].fillna(0).astype(int)
df['browse'] = df['browse'].fillna(1).astype(int)

# 2.5 标签：购买次数 ≥ 2 为回头客
df['is_returning'] = (df['purchases'] >= 2).astype(int)

# 2.6 补充特征
df['days_since'] = 30.0
df['cross_rate'] = 0.2

print(f"   合并后用户数: {len(df):,}")
print(f"   回头客占比: {df['is_returning'].mean()*100:.2f}%")

# ========== 3. 平衡采样 ==========
print("\n⚖️ 样本均衡处理...")

pos = df[df['is_returning'] == 1]
neg = df[df['is_returning'] == 0]

if len(pos) == 0:
    print("❌ 正样本数量为0，请检查数据")
    exit(1)

neg_sampled = neg.sample(n=min(len(neg), len(pos) * 10), random_state=42)
df_balanced = pd.concat([pos, neg_sampled]).sample(frac=1, random_state=42)

print(f"   平衡后样本: {len(df_balanced):,} (正:{len(pos)}, 负:{len(neg_sampled)})")

# ========== 4. 导出训练数据 ==========
X = df_balanced[['actions', 'browse', 'purchases', 'days_since', 'cross_rate']].values
y = df_balanced['is_returning'].values

np.savez('training_data.npz', X=X, y=y)

print(f"\n✅ 训练数据已导出: training_data.npz")
print(f"   X shape: {X.shape}")
print(f"   正样本: {y.sum()}, 负样本: {len(y) - y.sum()}")
print(f"\n💡 下一步: 运行 python train_lightgbm.py 开始训练")