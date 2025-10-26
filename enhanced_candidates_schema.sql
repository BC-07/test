-- Enhanced Candidates Table Schema
-- This enhances the existing candidates table to support real PDS extraction

-- First, let's see what we're working with (for reference)
-- The existing candidates table likely has: id, name, email, phone, job_id, score, status, etc.

-- Add new columns to existing candidates table for PDS integration
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS pds_extracted_data TEXT; -- Full JSON from extraction
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS total_education_entries INTEGER DEFAULT 0;
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS total_work_positions INTEGER DEFAULT 0;
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS total_training_hours DECIMAL(10,2) DEFAULT 0;
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS years_total_experience DECIMAL(5,2) DEFAULT 0;
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS processing_type VARCHAR(50) DEFAULT 'resume';
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS extraction_status VARCHAR(50) DEFAULT 'pending';
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS uploaded_filename VARCHAR(255);

-- Assessment Integration Columns
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS latest_assessment_id INTEGER;
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS latest_total_score DECIMAL(5,2);
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS latest_percentage_score DECIMAL(5,2);
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS latest_recommendation VARCHAR(50);
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS assessment_date DATETIME;

-- Category Breakdown Columns (from our working assessment engine)
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS education_score DECIMAL(5,2) DEFAULT 0;
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS experience_score DECIMAL(5,2) DEFAULT 0;
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS training_score DECIMAL(5,2) DEFAULT 0;
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS eligibility_score DECIMAL(5,2) DEFAULT 0;
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS accomplishments_score DECIMAL(5,2) DEFAULT 0;

-- Metadata columns
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS extraction_error TEXT;
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS assessment_error TEXT;
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS upload_batch_id VARCHAR(100); -- Group bulk uploads

-- Create index for better performance
CREATE INDEX IF NOT EXISTS idx_candidates_processing_type ON candidates(processing_type);
CREATE INDEX IF NOT EXISTS idx_candidates_job_extraction ON candidates(job_id, extraction_status);
CREATE INDEX IF NOT EXISTS idx_candidates_batch ON candidates(upload_batch_id);

-- Create a separate table for detailed assessment history (optional)
CREATE TABLE IF NOT EXISTS candidate_assessments (
    id INTEGER PRIMARY KEY,
    candidate_id INTEGER,
    job_id INTEGER,
    assessment_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Scores
    total_score DECIMAL(5,2),
    percentage_score DECIMAL(5,2),
    max_possible_score DECIMAL(5,2) DEFAULT 85,
    
    -- Category breakdown
    education_score DECIMAL(5,2),
    experience_score DECIMAL(5,2),
    training_score DECIMAL(5,2),
    eligibility_score DECIMAL(5,2),
    accomplishments_score DECIMAL(5,2),
    
    -- Results
    recommendation VARCHAR(50),
    assessment_details TEXT, -- JSON with full details
    engine_version VARCHAR(50) DEFAULT 'UniversityAssessmentEngine_v1',
    
    FOREIGN KEY (candidate_id) REFERENCES candidates(id),
    FOREIGN KEY (job_id) REFERENCES lspu_job_postings(id)
);

-- Migration script to clean old data (as requested)
-- WARNING: This will delete all existing candidates!
-- DELETE FROM candidates WHERE processing_type != 'real_pds_extraction' OR processing_type IS NULL;

-- Reset auto increment if needed
-- DELETE FROM sqlite_sequence WHERE name='candidates';