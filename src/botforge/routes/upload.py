from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from botforge.core.database import get_db
from botforge.services.uploader import UpstashVectorUploader
from botforge.repository.background_job import create_background_job, update_background_job_status

router = APIRouter()

async def background_upload(user_id: str, bot_id: str, job_id: str, db: AsyncSession):
    try:
        await update_background_job_status(db, job_id, status="running", start=True)

        uploader = UpstashVectorUploader()
        result = uploader.upload_user_bot_data(user_id, bot_id)

        await update_background_job_status(db, job_id, status="success", result={"details": result}, finish=True)
    except Exception as e:
        await update_background_job_status(
            db, job_id, status="failed", result={"error": str(e)}, finish=True, increment_attempt=True
        )


@router.post("", tags=["Upload"])
async def upload_user_bot_data(
    user_id: str,
    bot_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    job = await create_background_job(db, job_type="upload_user_bot_data", user_id=user_id, bot_id=bot_id)
    background_tasks.add_task(background_upload, user_id, bot_id, str(job.id), db)
    return {"message": "Upload started in background", "job_id": str(job.id)}
