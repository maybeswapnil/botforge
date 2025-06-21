from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    upstash_url: str
    upstash_token: str

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int

    class Config:
        env_file = ".env"

    @property
    def postgres_uri(self):
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

settings = Settings()
