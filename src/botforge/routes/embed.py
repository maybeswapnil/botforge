from fastapi import APIRouter
from pydantic import BaseModel
from botforge.core.vector_model import model

router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/embed")
def embed(input: TextInput):
    embedding = model.encode(input.text).tolist()
    return {"embedding": embedding}
