from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    upstash_url: str
    upstash_token: str

    upload_location: str
    
    openai_api_key: str
    openai_default_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 1000
    openai_temperature: float = 0.7
    
    default_top_k: int = 5
    max_top_k: int = 20

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int

    redis_uri: str = "redis://localhost:6379/0"
    default_history_size: int = 3

    class Config:
        env_file = ".env"

    @property
    def postgres_uri(self):
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

settings = Settings()
