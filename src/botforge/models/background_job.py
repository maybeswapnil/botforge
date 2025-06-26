""" Background job model """

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, DateTime, JSON, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class BackgroundJob(Base):
    __tablename__ = "background_jobs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    job_type = Column(String(100), nullable=False)
    user_id = Column(String(100), nullable=True)
    bot_id = Column(String(100), nullable=True)
    status = Column(String(20), nullable=False, default="queued")
    attempts = Column(Integer, nullable=False, default=0)
    result = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
