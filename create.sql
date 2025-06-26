CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- For gen_random_uuid()

CREATE TABLE background_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type VARCHAR(100) NOT NULL,                 -- e.g., 'upload_user_bot_data'
    user_id VARCHAR(100),                           -- Optional association
    bot_id VARCHAR(100),                            -- Optional association
    status VARCHAR(20) NOT NULL DEFAULT 'queued',   -- queued, running, success, failed
    attempts INTEGER NOT NULL DEFAULT 0,
    result JSONB,                                   -- Store result or error info
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

-- Optional: Add indexes for faster querying
CREATE INDEX idx_background_jobs_status ON background_jobs(status);
CREATE INDEX idx_background_jobs_user_id ON background_jobs(user_id);
CREATE INDEX idx_background_jobs_bot_id ON background_jobs(bot_id);