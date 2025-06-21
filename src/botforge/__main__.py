from botforge.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("botforge.main:app", host="0.0.0.0", port=8000, reload=True)
