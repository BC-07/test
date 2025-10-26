-- Create database (run this separately in PostgreSQL)
-- CREATE DATABASE resumai_db;

-- Job categories table
CREATE TABLE IF NOT EXISTS job_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id SERIAL PRIMARY KEY,
    title VARCHAR(100) NOT NULL,
    department VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    requirements TEXT NOT NULL,
    experience_level VARCHAR(20) NOT NULL DEFAULT 'mid',
    category_id INTEGER REFERENCES job_categories(id) ON DELETE SET NULL,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Candidates table
CREATE TABLE IF NOT EXISTS candidates (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(120),
    phone VARCHAR(20),
    linkedin VARCHAR(200),
    github VARCHAR(200),
    resume_text TEXT NOT NULL,
    category VARCHAR(50),
    skills TEXT,
    education JSONB,
    experience JSONB,
    status VARCHAR(20) DEFAULT 'new',
    score FLOAT DEFAULT 0.0,
    job_id INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
    notes TEXT,
    -- PDS-specific fields
    pds_data JSONB,  -- Store comprehensive PDS information
    certifications JSONB,  -- Store certifications separately
    training JSONB,  -- Store training/seminars
    awards JSONB,  -- Store awards and recognition
    eligibility JSONB,  -- Store civil service eligibility
    languages JSONB,  -- Store language proficiency
    licenses JSONB,  -- Store professional licenses
    volunteer_work JSONB,  -- Store volunteer activities
    personal_references JSONB,  -- Store personal references
    government_ids JSONB,  -- Store government ID numbers
    scoring_breakdown JSONB,  -- Store detailed scoring breakdown
    processing_type VARCHAR(20) DEFAULT 'resume',  -- 'resume' or 'pds'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scoring criteria configuration table
CREATE TABLE IF NOT EXISTS scoring_criteria (
    id SERIAL PRIMARY KEY,
    criteria_name VARCHAR(50) NOT NULL,
    criteria_config JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analytics table
CREATE TABLE IF NOT EXISTS analytics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    total_resumes INTEGER DEFAULT 0,
    processed_resumes INTEGER DEFAULT 0,
    shortlisted INTEGER DEFAULT 0,
    rejected INTEGER DEFAULT 0,
    avg_processing_time FLOAT DEFAULT 0.0,
    job_category_stats JSONB
);

-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Settings table
CREATE TABLE IF NOT EXISTS settings (
    id SERIAL PRIMARY KEY,
    key VARCHAR(50) UNIQUE NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_candidates_job_id ON candidates(job_id);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);
CREATE INDEX IF NOT EXISTS idx_candidates_score ON candidates(score);
CREATE INDEX IF NOT EXISTS idx_candidates_processing_type ON candidates(processing_type);
CREATE INDEX IF NOT EXISTS idx_jobs_category_id ON jobs(category_id);
CREATE INDEX IF NOT EXISTS idx_analytics_date ON analytics(date);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_scoring_criteria_active ON scoring_criteria(is_active);

-- Insert default admin user (password: admin123)
INSERT INTO users (email, password_hash, first_name, last_name, is_admin)
VALUES ('admin@resumeai.com', '$2b$12$P9/d224UJ3fGh3rbTjRiWeIkehLv3QvNn5vweGK6SThKoOSfi9E7C', 'Admin', 'User', TRUE)
ON CONFLICT (email) DO NOTHING;

-- Insert default job category if none exists
INSERT INTO job_categories (name, description) 
VALUES ('Software Development', 'Software development and engineering roles')
ON CONFLICT (name) DO NOTHING;

-- Insert default scoring criteria
INSERT INTO scoring_criteria (criteria_name, criteria_config, is_active)
VALUES (
    'default_pds_scoring',
    '{"education": {"weight": 0.25, "subcriteria": {"relevance": 0.4, "level": 0.3, "institution": 0.2, "grades": 0.1}}, "experience": {"weight": 0.30, "subcriteria": {"relevance": 0.5, "duration": 0.3, "responsibilities": 0.2}}, "skills": {"weight": 0.20, "subcriteria": {"technical_match": 0.6, "certifications": 0.4}}, "personal_attributes": {"weight": 0.15, "subcriteria": {"eligibility": 0.5, "awards": 0.3, "training": 0.2}}, "additional_qualifications": {"weight": 0.10, "subcriteria": {"languages": 0.4, "licenses": 0.3, "volunteer_work": 0.3}}}',
    TRUE
)
ON CONFLICT DO NOTHING;

-- Insert default job if none exists
INSERT INTO jobs (title, department, description, requirements, category_id)
SELECT 
    'Software Developer',
    'Engineering',
    'We are looking for a skilled software developer to join our team.',
    'Python, JavaScript, React, Node.js, SQL, Git',
    (SELECT id FROM job_categories WHERE name = 'Software Development')
WHERE NOT EXISTS (SELECT 1 FROM jobs);