from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import lightgbm as lgb
import numpy as np
import uvicorn
import math

app = FastAPI(title="复购预测API")

# 允许前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载模型
print("加载模型中...")
model = lgb.Booster(model_file='lightgbm_model.txt')
print("模型加载成功")
print(f"模型特征数量: {model.num_feature()}")

class PredictRequest(BaseModel):
    actions: float
    browse: float
    purchases: float

class PredictResponse(BaseModel):
    probability: float
    strategy: str

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(req: PredictRequest):
    # 构造特征（与训练时一致，需要5个特征）
    # 特征顺序: actions, browse, purchases, days_since, cross_rate
    features = np.array([[
        req.actions,      # 行为次数
        req.browse,       # 浏览商品数
        req.purchases,    # 历史购买次数
        30.0,             # days_since（最近一次购买天数，默认30天）
        0.2               # cross_rate（跨类目购买率，默认0.2）
    ]])
    
    # 获取原始预测值（raw score，范围约 -4.5 到 4.4）
    raw_score = model.predict(features)[0]
    
    # 通过 sigmoid 函数转换为概率
    # sigmoid(x) = 1 / (1 + e^(-x))
    prob = 1.0 / (1.0 + math.exp(-raw_score))
    
    # 确保概率在合理范围内（1% - 99%）
    prob = max(0.01, min(0.99, prob))
    
    # 根据概率确定策略
    if prob > 0.7:
        strategy = "立即唤醒"
    elif prob > 0.4:
        strategy = "适时引导"
    else:
        strategy = "长期培育"
    
    # 打印日志（方便调试）
    print(f"输入: act={req.actions}, brw={req.browse}, pur={req.purchases}")
    print(f"原始分: {raw_score:.4f} → 概率: {prob*100:.2f}% → 策略: {strategy}")
    
    return {
        "probability": round(prob, 3),
        "strategy": strategy
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)