from fastapi import FastAPI, HTTPException
from typing import List, Dict, Optional
import uvicorn
from datetime import datetime

from .schemas import (
    Transaction, AssociationRule, FrequentItemset, 
    AnalysisRequest, RecommendationRequest, 
    BasketRecommendation, AlgorithmType
)
from .model import MarketBasketAnalyzer

app = FastAPI(
    title="Market Basket Analysis API",
    description="API for market basket analysis using association rules",
    version="1.0.0"
)

analyzer = MarketBasketAnalyzer()

@app.get("/")
async def root():
    return {"message": "Market Basket Analysis API", "status": "active"}

@app.post("/analyze")
async def analyze_baskets(request: AnalysisRequest):
    try:
        analyzer.load_transactions(request.transactions)
        analyzer.analyze(min_support=request.min_support)
        
        return {
            "frequent_itemsets": analyzer.frequent_itemsets,
            "association_rules": [],
            "total_transactions": len(request.transactions),
            "min_support": request.min_support,
            "min_confidence": request.min_confidence,
            "algorithm": request.algorithm
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")

@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    try:
        recommendations = analyzer.get_recommendations(
            items=request.current_items,
            top_k=request.max_recommendations
        )
        
        return {
            "recommendations": recommendations,
            "input_items": request.current_items,
            "recommendation_count": len(recommendations)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Recommendation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)