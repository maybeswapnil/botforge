from sentence_transformers import SentenceTransformer
from upstash_vector import Index
from botforge.core.config import settings

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

index = Index(
    url=settings.upstash_url,
    token=settings.upstash_token
)
