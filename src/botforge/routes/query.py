from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.botforge.services.vector_query import VectorQueryService
from botforge.core.logger import log

router = APIRouter(prefix="", tags=["Vector Queries"])

# Request/Response Models
class QueryRequest(BaseModel):
    user_id: str
    bot_id: str
    query: str
    model: str = "gpt-3.5-turbo"  # Default OpenAI model
    top_k: int = 5
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7

class QueryResponse(BaseModel):
    user_id: str
    bot_id: str
    query: str
    model: str
    response: str
    sources: List[Dict[str, Any]]
    total_chunks_found: int

class HealthResponse(BaseModel):
    status: str
    upstash_connection: bool
    openai_connection: bool
    timestamp: str

class TestResponse(BaseModel):
    status: str
    test_query: str
    response: str
    execution_time: float

# Dependency to get service instance
def get_vector_service() -> VectorQueryService:
    return VectorQueryService()

@router.post("/query", response_model=QueryResponse)
async def query_vector_data(
    request: QueryRequest,
    service: VectorQueryService = Depends(get_vector_service)
):
    """
    Query vector database and generate response using OpenAI model
    """
    try:
        log.info(f"Processing query for user: {request.user_id}, bot: {request.bot_id}")
        
        result = await service.query_and_generate_response(
            user_id=request.user_id,
            bot_id=request.bot_id,
            query=request.query,
            model=request.model,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return QueryResponse(
            user_id=request.user_id,
            bot_id=request.bot_id,
            query=request.query,
            model=request.model,
            response=result["response"],
            sources=result["sources"],
            total_chunks_found=result["total_chunks_found"]
        )
        
    except Exception as e:
        log.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/health", response_model=HealthResponse)
async def health_check(service: VectorQueryService = Depends(get_vector_service)):
    """
    Check health of vector database and OpenAI connections
    """
    try:
        health_status = await service.health_check()
        
        return HealthResponse(
            status="healthy" if health_status["overall_status"] else "unhealthy",
            upstash_connection=health_status["upstash_status"],
            openai_connection=health_status["openai_status"],
            timestamp=health_status["timestamp"]
        )
        
    except Exception as e:
        log.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.post("/test", response_model=TestResponse)
async def test_query(service: VectorQueryService = Depends(get_vector_service)):
    """
    Test endpoint with sample query
    """
    try:
        test_result = await service.test_query()
        
        return TestResponse(
            status="success",
            test_query=test_result["test_query"],
            response=test_result["response"],
            execution_time=test_result["execution_time"]
        )
        
    except Exception as e:
        log.error(f"Test query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@router.get("/namespaces")
async def list_namespaces(service: VectorQueryService = Depends(get_vector_service)):
    """
    List available namespaces (user/bot combinations)
    """
    try:
        namespaces = await service.list_available_namespaces()
        return {"namespaces": namespaces}
        
    except Exception as e:
        log.error(f"Failed to list namespaces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list namespaces: {str(e)}")

@router.get("/stats/{user_id}/{bot_id}")
async def get_namespace_stats(
    user_id: str, 
    bot_id: str,
    service: VectorQueryService = Depends(get_vector_service)
):
    """
    Get statistics for a specific user/bot namespace
    """
    try:
        stats = await service.get_namespace_stats(user_id, bot_id)
        return stats
        
    except Exception as e:
        log.error(f"Failed to get stats for {user_id}/{bot_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")