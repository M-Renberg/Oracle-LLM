from pydantic import BaseModel, Field
from typing import List, Dict

class UploadMetadataResponse(BaseModel):
    rows: int
    columns: List[str]
    dtypes: Dict[str, str]

class AskRequest(BaseModel):
    question: str = Field(..., description="Question about the data")

class AskResponse(BaseModel):
    question: str
    answer: str
    model: str