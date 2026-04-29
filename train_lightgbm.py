"""LightGBM 复购预测模型训练脚本"""
import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import os

print("=" * 50)
print("LightGBM 复购预测模型训练")
print("=" * 50)

# 读取 JSON 数据
try:
    with open('output/repurchase_analysis_data.json', 'r', encoding='utf-8') as f:
        rep_data = json.load(f)
    print("✅ 成功加载数据")
    print(f"   总用户数: {rep_data['stats']['total_users']:,}")
    print(f"   回头客数: {rep_data['stats']['returning_users']:,}")
    print(f"   回头率: {rep_data['stats']['returning_rate']}%")
except:
    print("⚠️ 无法加载 JSON，使用模拟数据")
    rep_data = None

# ============================================================
# 优先加载真实预处理数据
# 如果已通过 preprocess_for_training.py 生成训练数据，直接使用
# ============================================================
if os.path.exists('training_data.npz'):
    print("✅ 检测到真实预处理数据，加载中...")
    real_data = np.load('training_data.npz')
    X = real_data['X']
    y = real_data['y']
    print(f"   训练样本: {len(X)} 条")
    print(f"   回头客占比: {y.mean()*100:.2f}%")
else:
    print("⚠️ 未找到预处理数据(training_data.npz)，使用模拟数据演示")
    print("   如需使用真实数据，请先运行: python preprocess_for_training.py")
    print("")

    # ========模拟数据生成代码 ========

    # 生成训练样本
    np.random.seed(42)
    n_samples = 5000

    # 特征：行为次数、浏览商品数、历史购买次数、距上次购买天数、跨类目比例
    X = []
    y = []

    # 回头率约 1.14%，正负样本严重不均衡
    returning_rate = 0.0114
    n_returning = int(n_samples * returning_rate)
    n_non_returning = n_samples - n_returning

    print(f"生成训练样本...")
    print(f"   正样本（回头客）: {n_returning}")
    print(f"   负样本（非回头客）: {n_non_returning}")

    # 生成正样本（回头客）
    for _ in range(n_returning):
        actions = np.random.randint(100, 300)
        browse = np.random.randint(20, 60)
        purchases = np.random.choice([2, 3, 4, 5], p=[0.6, 0.25, 0.1, 0.05])
        days_since = np.random.randint(1, 30)
        cross_rate = np.random.uniform(0.1, 0.5)
        X.append([actions, browse, purchases, days_since, cross_rate])
        y.append(1)

    # 生成负样本（非回头客）
    for _ in range(n_non_returning):
        actions = np.random.randint(5, 80)
        browse = np.random.randint(1, 20)
        purchases = np.random.choice([0, 0, 1], p=[0.7, 0.2, 0.1])
        days_since = np.random.randint(30, 180)
        cross_rate = np.random.uniform(0, 0.15)
        X.append([actions, browse, purchases, days_since, cross_rate])
        y.append(0)

    # ======== 模拟数据生成代码结束 ========

# 转换为 numpy 数组
X = np.array(X)
y = np.array(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")

# 训练 LightGBM 模型
print("\n开始训练 LightGBM 模型...")

# 正负样本比例严重失衡，设置 scale_pos_weight
if os.path.exists('training_data.npz'):
    scale_pos_weight = (len(y) - y.sum()) / y.sum()
else:
    scale_pos_weight = n_non_returning / n_returning

model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n✅ 模型训练完成")
print(f"   准确率: {accuracy:.4f}")
print(f"   召回率: {recall:.4f}")

# 保存模型
model.booster_.save_model('lightgbm_model.txt')
print(f"\n✅ 模型已保存: lightgbm_model.txt")

# 特征重要性
print("\n特征重要性:")
features = ['行为次数', '浏览商品数', '历史购买', '距上次购买', '跨类目比例']
importance = model.feature_importances_
for name, imp in zip(features, importance):
    print(f"   {name}: {imp}")