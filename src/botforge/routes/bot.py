from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/bots")
async def list_bots():
    # Sample response
    return [{"id": 1, "name": "Bot Alpha"}, {"id": 2, "name": "Bot Beta"}]

@router.get("/bots/{bot_id}")
async def get_bot(bot_id: int):
    # Sample logic
    if bot_id == 1:
        return {"id": 1, "name": "Bot Alpha"}
    elif bot_id == 2:
        return {"id": 2, "name": "Bot Beta"}
    else:
        raise HTTPException(status_code=404, detail="Bot not found")

@router.post("/bots")
async def create_bot(bot: dict):
    # Sample creation logic
    return {"id": 3, "name": bot.get("name", "Unnamed Bot")}
