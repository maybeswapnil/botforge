# botforge/api/routes/upload.py

from fastapi import APIRouter
from botforge.services.uploader import UpstashVectorUploader

router = APIRouter()

@router.post("", tags=["Upload"])
def upload_user_bot_data(user_id: str, bot_id: str):
    uploader = UpstashVectorUploader()
    result = uploader.upload_user_bot_data(user_id, bot_id)
    return {"message": "Upload complete", "details": result}
