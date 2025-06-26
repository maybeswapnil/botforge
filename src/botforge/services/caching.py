import json
from redis.asyncio.client import Redis
from typing import Optional
from botforge.core.config import settings

async def store_query_response_context(
    redis: Redis,
    client_id: str,
    query: str,
    response: str,
    max_context_length: Optional[int] = None
):
    """
    Stores the latest query-response pair in Redis under the user's context key.

    Args:
        redis (Redis): The Redis client instance.
        client_id (str): Unique client identifier.
        query (str): User query.
        response (str): RAG model response.
        max_context_length (int, optional): Max history items to retain per user.
                                          Defaults to env var default_history_size or 3.
    """
    # Get max context length from parameter, env var, or default
    if max_context_length is None:
        max_context_length = settings.default_history_size
    
    key = f"user_context:{client_id}"
    
    # Get existing context list
    context = await redis.get(key)
    if context:
        try:
            context_list = json.loads(context)
        except json.JSONDecodeError:
            context_list = []
    else:
        context_list = []

    # If at max capacity, remove oldest entry
    if len(context_list) >= max_context_length:
        context_list.pop(0)  # Remove oldest (first) entry
    
    # Append new query-response pair
    context_list.append({"query": query, "response": response})

    # Save back to Redis
    await redis.set(key, json.dumps(context_list), ex=60 * 1)  # 1 minute expiry

async def get_query_response_context(redis: Redis, client_id: str):
    """
    Retrieves the query-response context for a user.

    Args:
        redis (Redis): The Redis client instance.
        client_id (str): Unique client identifier.

    Returns:
        List of query-response pairs or empty list if not found.
    """
    key = f"user_context:{client_id}"
    context = await redis.get(key)
    
    if context:
        try:
            return json.loads(context)
        except json.JSONDecodeError:
            return []
    
    return []  # No context found

async def get_context_length(redis: Redis, client_id: str):
    """
    Get the current number of context entries for a user.
    
    Args:
        redis (Redis): The Redis client instance.
        client_id (str): Unique client identifier.
    
    Returns:
        int: Number of context entries stored for the user.
    """
    context_list = await get_query_response_context(redis, client_id)
    return len(context_list)