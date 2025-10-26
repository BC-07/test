-- Upload Session Management Table
CREATE TABLE IF NOT EXISTS upload_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    user_id INTEGER,
    job_id INTEGER NOT NULL,
    status TEXT DEFAULT 'pending',
    file_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    session_data TEXT, -- JSON data for file information
    error_log TEXT,
    FOREIGN KEY (job_id) REFERENCES lspu_job_postings (id)
);

-- Upload Files tracking table
CREATE TABLE IF NOT EXISTS upload_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    file_id TEXT UNIQUE NOT NULL,
    original_name TEXT NOT NULL,
    temp_path TEXT NOT NULL,
    file_size INTEGER,
    file_type TEXT,
    status TEXT DEFAULT 'uploaded',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    candidate_id INTEGER,
    error_message TEXT,
    FOREIGN KEY (session_id) REFERENCES upload_sessions (session_id),
    FOREIGN KEY (candidate_id) REFERENCES candidates (id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_upload_sessions_user_id ON upload_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_upload_sessions_job_id ON upload_sessions(job_id);
CREATE INDEX IF NOT EXISTS idx_upload_sessions_status ON upload_sessions(status);
CREATE INDEX IF NOT EXISTS idx_upload_files_session_id ON upload_files(session_id);
CREATE INDEX IF NOT EXISTS idx_upload_files_status ON upload_files(status);