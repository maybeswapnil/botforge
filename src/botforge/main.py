from fastapi import FastAPI
from botforge.routes import embed, query, upload
from botforge.core.logger import log

app = FastAPI()
log.info("BotForge API initialized")

app.include_router(embed.router, prefix="/embed", tags=["Embedding"])
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(upload.router, prefix="/upload", tags=["Upload"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
