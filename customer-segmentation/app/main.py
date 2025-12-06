from fastapi import FastAPI
from app.schemas import CustomerRequest, CustomerResponse
from app.model import load_model, ensure_db, predict_cluster, save_result, get_results, get_cluster_summary

app = FastAPI(title="Customer Segmentation API")

@app.on_event("startup")
async def startup_event():
    load_model()
    ensure_db()

@app.get("/")
def root():
    return {"message": "Customer Segmentation API"}

@app.post("/segment", response_model=CustomerResponse)
def segment(req: CustomerRequest):
    c = predict_cluster(req)
    rid = save_result(req, c)
    return CustomerResponse(cluster=c, id=rid)

@app.get("/results")
def results(limit: int = 50):
    rows = get_results(limit)
    return {"results": rows}

@app.get("/cluster-summary")
def cluster_summary():
    rows = get_cluster_summary()
    return {"summary": rows}
