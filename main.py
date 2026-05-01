from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import lightgbm as lgb
import numpy as np
import uvicorn
import math

app = FastAPI(title="复购预测API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    features = np.array([[
        req.actions, req.browse, req.purchases, 30.0, 0.2
    ]])
    raw_score = model.predict(features)[0]
    model_prob = 1.0 / (1.0 + math.exp(-raw_score))
    actions_score = min(req.actions / 200, 1)
    browse_score = min(req.browse / 40, 1)
    purchases_score = min(req.purchases / 3, 1)
    rule_prob = purchases_score * 0.6 + browse_score * 0.25 + actions_score * 0.15
    prob = model_prob * 0.3 + rule_prob * 0.7
    prob = max(0.05, min(0.90, prob))
    if prob > 0.7:
        strategy = "立即唤醒"
    elif prob > 0.4:
        strategy = "适时引导"
    else:
        strategy = "长期培育"
    print(f"模型: {model_prob*100:.1f}% | 规则: {rule_prob*100:.1f}% | 混合: {prob*100:.2f}%")
    return {
        "probability": round(prob, 3),
        "strategy": strategy
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
