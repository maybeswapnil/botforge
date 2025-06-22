# botforge/api/routes/query.py

from fastapi import APIRouter, Query
from botforge.services.query_engine import VectorSearchQA

router = APIRouter()

@router.post("/")
def query_qa(question: str = Query(..., min_length=5), user_id: str = "", bot_id: str = ""):
    engine = VectorSearchQA(user_id=user_id, bot_id=bot_id)
    answer = engine.answer(question)
    return {
        "question": question,
        "answer": answer
    }
