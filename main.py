# app/main.py
from fastapi import FastAPI, UploadFile, HTTPException
from datahandler import load_csv_to_memory, get_dataframe, get_stats
from chain.pipline import run_oracle
from schemas import AskRequest, AskResponse, UploadMetadataResponse
from chain.llm import MODELS
import pandas as pd

app = FastAPI()

STOPWORDS = {
    "what", "which", "that", "have", "sold", "most", "units", "game",
    "best", "many", "does", "with", "from", "this", "their", "when",
    "who", "the", "and", "for", "are", "has", "been", "were", "will",
}


@app.get("/")
async def root():
    return {"message": "Oraklet-API är igång! Gå till /docs för att testa endpoints."}

@app.get("/ai/models")
async def list_models():
    return {"available_models": list(MODELS.keys())}


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
async def ask_question(request: AskRequest, model: str = "smollm2"):
    if model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Okänd modell '{model}'. Tillgängliga: {list(MODELS.keys())}")
 
    stats = get_stats()
    df = get_dataframe()
 
    words = [w for w in request.question.lower().split()
             if len(w) > 3 and w not in STOPWORDS]
 
    mask = pd.Series([False] * len(df), index=df.index)
    for word in words:
        mask |= df.apply(
            lambda row: row.astype(str).str.lower().str.contains(word, regex=False).any(), axis=1
        )
 
    filtered = df[mask] if mask.any() else df
 
    numeric_cols = filtered.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        filtered = filtered.sort_values(numeric_cols[-1], ascending=False)
 
    context = {"summary": stats, "head": filtered.head(10)}
 
    try:
        return run_oracle(request.question, context, model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
