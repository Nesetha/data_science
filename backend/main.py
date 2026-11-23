from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag.retrieval import answer_question
from src.safety import validate_question

app = FastAPI(title="PrenatalAI API", description="Research prototype; not for diagnosis.", version="0.1.0")

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

@app.get("/")
def root():
    return {"application": "PrenatalAI", "status": "running", "clinical_use": "research_only"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/assistant/query")
def query(req: QuestionRequest):
    valid, reason = validate_question(req.question)
    if not valid:
        raise HTTPException(status_code=400, detail=reason)
    return answer_question(req.question)
