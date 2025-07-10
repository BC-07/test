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
CREATE INDEX IF NOT EXISTS idx_jobs_category_id ON jobs(category_id);
CREATE INDEX IF NOT EXISTS idx_analytics_date ON analytics(date);

-- Insert default job category if none exists
INSERT INTO job_categories (name, description) 
VALUES ('Software Development', 'Software development and engineering roles')
ON CONFLICT (name) DO NOTHING;

-- Insert default job if none exists
INSERT INTO jobs (title, department, description, requirements, category_id)
SELECT 
    'Software Developer',
    'Engineering',
    'We are looking for a skilled software developer to join our team.',
    'Python, JavaScript, React, Node.js, SQL, Git',
    (SELECT id FROM job_categories WHERE name = 'Software Development')
WHERE NOT EXISTS (SELECT 1 FROM jobs);