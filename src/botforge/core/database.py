from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from botforge.core.config import settings

DATABASE_URL = settings.postgres_uri

engine = create_async_engine(DATABASE_URL, echo=False)

AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

# Dependency for routes
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
