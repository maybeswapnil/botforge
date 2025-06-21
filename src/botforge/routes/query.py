import base64
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from botforge.core.vector_model import index

router = APIRouter()

class QueryInput(BaseModel):
    data: str
    top_k: int = 1
    include_vectors: bool = True
    include_metadata: bool = True

@router.post("/query")
def query(input: QueryInput):
    result = index.query(
        data=input.data,
        top_k=input.top_k,
        include_vectors=input.include_vectors,
        include_metadata=input.include_metadata,
    )
    if "matches" in result:
        for match in result["matches"]:
            if "vector" in match and match["vector"] is not None:
                decoded = base64.b64decode(match["vector"])
                match["vector"] = np.frombuffer(decoded, dtype=np.float32).tolist()
    return result
