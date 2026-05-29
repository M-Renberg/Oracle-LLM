# app/main.py
from fastapi import FastAPI, UploadFile, HTTPException
from datahandler import load_csv_to_memory, get_dataframe, get_stats
from chain.pipline import run_oracle
from schemas import AskRequest, AskResponse, UploadMetadataResponse

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Oraklet-API är igång! Gå till /docs för att testa endpoints."}

@app.post("/data/upload")
async def upload_data(file: UploadFile):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Filen måste vara en CSV")
    
    content = await file.read()
    df = load_csv_to_memory(content)
    
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

@app.post("/ai/ask")
async def ask_question(request: AskRequest):
    stats = get_stats() 
    
    try:
        return run_oracle(request.question, stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))