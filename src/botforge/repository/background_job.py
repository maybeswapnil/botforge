""" Background job repository """

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from botforge.models.background_job import BackgroundJob
from typing import Optional, Dict
from datetime import datetime

async def update_background_job_status(
    db: AsyncSession,
    job_id,
    status: str,
    result: Optional[Dict] = None,
    start: bool = False,
    finish: bool = False,
    increment_attempt: bool = False
) -> Optional[BackgroundJob]:
    stmt = select(BackgroundJob).where(BackgroundJob.id == job_id)
    result_obj = await db.execute(stmt)
    job = result_obj.scalar_one_or_none()

    if not job:
        return None

    job.status = status
    if result is not None:
        job.result = result
    if start:
        job.started_at = datetime.utcnow()
    if finish:
        job.finished_at = datetime.utcnow()
    if increment_attempt:
        job.attempts += 1

    await db.commit()
    await db.refresh(job)
    return job

async def create_background_job(
    db: AsyncSession,
    job_type: str,
    user_id: Optional[str] = None,
    bot_id: Optional[str] = None
) -> BackgroundJob:
    job = BackgroundJob(
        job_type=job_type,
        user_id=user_id,
        bot_id=bot_id,
        status='queued'
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job

