import redis.asyncio as redis
from botforge.core.config import settings

REDIS_URL = settings.redis_uri

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,     # decode bytes to str automatically
    socket_timeout=5,          # short timeout for network issues
    socket_connect_timeout=5,  # connection timeout
    health_check_interval=30,  # periodically ping to keep connection alive
    max_connections=10         # connection pool size
)

async def get_redis():
    try:
        yield redis_client
    finally:
        await redis_client.close()