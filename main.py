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
    # 对高购买次数进行衰减处理
    # 购买次数>2时，对数化处理，降低极端值的影响
    purchases_capped = req.purchases if req.purchases <= 1 else 1 + math.log2(req.purchases)
    
    features = np.array([[
        req.actions,
        req.browse,
        purchases_capped,  # 使用衰减后的值
        30.0,
        0.2
    ]])
    
    raw_score = model.predict(features)[0]
    prob = 1.0 / (1.0 + math.exp(-raw_score))
    prob = max(0.01, min(0.99, prob))
    
    if prob > 0.7:
        strategy = "立即唤醒"
    elif prob > 0.4:
        strategy = "适时引导"
    else:
        strategy = "长期培育"
    
    print(f"输入: act={req.actions}, brw={req.browse}, pur={req.purchases}→{purchases_capped:.2f}")
    print(f"原始分: {raw_score:.4f} → 概率: {prob*100:.2f}% → 策略: {strategy}")
    
    return {
        "probability": round(prob, 3),
        "strategy": strategy
    }
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
