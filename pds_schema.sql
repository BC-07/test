-- PDS Candidates table - completely separate from regular candidates
CREATE TABLE IF NOT EXISTS pds_candidates (
    id SERIAL PRIMARY KEY,
    
    -- Basic Information
    name VARCHAR(100) NOT NULL,
    email VARCHAR(120),
    phone VARCHAR(20),
    
    -- Job Information
    job_id INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
    score FLOAT DEFAULT 0.0,
    status VARCHAR(20) DEFAULT 'new',
    
    -- File Information
    filename VARCHAR(255) NOT NULL,
    file_size INTEGER,
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Core PDS Data (JSONB for flexibility)
    personal_info JSONB NOT NULL,        -- Name, address, contact details, civil status, etc.
    family_background JSONB,             -- Spouse, children, parents info
    educational_background JSONB,        -- Elementary, secondary, vocational, college, graduate studies
    civil_service_eligibility JSONB,     -- CS exams, ratings, dates, places
    work_experience JSONB,               -- Position, company, salary, dates, govt service status
    voluntary_work JSONB,                -- Organization, position, dates, hours
    learning_development JSONB,          -- Training programs, seminars, dates, hours, type
    other_information JSONB,             -- Special skills, recognition, membership, etc.
    personal_references JSONB,           -- Name, position, company, contact details
    government_ids JSONB,                -- SSS, PagIBIG, PhilHealth, TIN, etc.
    
    -- Extracted Summary Data for Quick Access
    highest_education VARCHAR(100),
    years_of_experience INTEGER DEFAULT 0,
    government_service_years INTEGER DEFAULT 0,
    civil_service_eligible BOOLEAN DEFAULT FALSE,
    
    -- Scoring Details
    scoring_breakdown JSONB,             -- Detailed breakdown of how score was calculated
    matched_qualifications JSONB,       -- Specific qualifications that matched job requirements
    areas_for_improvement JSONB,        -- Areas where candidate could improve
    
    -- Processing Metadata
    extraction_success BOOLEAN DEFAULT TRUE,
    extraction_errors JSONB,            -- Any errors during PDS extraction
    processing_notes TEXT,               -- Additional notes about processing
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_pds_candidates_job_id ON pds_candidates(job_id);
CREATE INDEX IF NOT EXISTS idx_pds_candidates_status ON pds_candidates(status);
CREATE INDEX IF NOT EXISTS idx_pds_candidates_score ON pds_candidates(score);
CREATE INDEX IF NOT EXISTS idx_pds_candidates_email ON pds_candidates(email);
CREATE INDEX IF NOT EXISTS idx_pds_candidates_upload_timestamp ON pds_candidates(upload_timestamp);
CREATE INDEX IF NOT EXISTS idx_pds_candidates_civil_service ON pds_candidates(civil_service_eligible);

-- Create update trigger for updated_at
CREATE OR REPLACE FUNCTION update_pds_candidates_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_pds_candidates_updated_at
    BEFORE UPDATE ON pds_candidates
    FOR EACH ROW
    EXECUTE FUNCTION update_pds_candidates_updated_at();