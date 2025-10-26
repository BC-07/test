-- LSPU University Job Posting System Database Schema
-- Enhanced schema to support LSPU-style job postings with all required fields

-- University configuration table for branding and settings
CREATE TABLE IF NOT EXISTS university_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    university_name VARCHAR(200) DEFAULT 'Laguna State Polytechnic University',
    university_logo_url TEXT,
    primary_color VARCHAR(7) DEFAULT '#1e3a8a', -- Blue color from images
    secondary_color VARCHAR(7) DEFAULT '#10b981', -- Green footer color
    contact_person_name VARCHAR(100) DEFAULT 'MARIO R. BRIONES, EdD',
    contact_person_title VARCHAR(100) DEFAULT 'University President',
    university_website VARCHAR(100) DEFAULT 'lspu.edu.ph',
    facebook_page VARCHAR(200) DEFAULT 'facebook.com/LSPUOfficial',
    hr_email VARCHAR(100) DEFAULT 'information.office@lspu.edu.ph',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Campus locations table
CREATE TABLE IF NOT EXISTS campus_locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campus_name VARCHAR(100) NOT NULL,
    campus_code VARCHAR(10),
    address TEXT,
    contact_email VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Enhanced job postings table with all LSPU-specific fields
CREATE TABLE IF NOT EXISTS lspu_job_postings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Basic Information
    job_reference_number VARCHAR(50) UNIQUE, -- e.g., "2025-LSPU-JOBS-093"
    position_title VARCHAR(200) NOT NULL, -- e.g., "FOUR (4) INSTRUCTORS"
    specific_role VARCHAR(200), -- e.g., "Certified Public Accountant (CPA)"
    quantity_needed INTEGER DEFAULT 1,
    
    -- Position Classification
    position_type_id INTEGER NOT NULL, -- Links to position_types table (1-4)
    position_category VARCHAR(50), -- "TEACHING", "NON-TEACHING", etc.
    campus_id INTEGER,
    department_office VARCHAR(200),
    
    -- Administrative Details
    plantilla_item_no VARCHAR(50), -- e.g., "LSPCB-ADOF4-105-2022"
    salary_grade INTEGER,
    salary_amount DECIMAL(12,2),
    employment_period VARCHAR(100), -- e.g., "First Semester, Academic Year 2025-2026"
    
    -- Qualifications
    education_requirements TEXT NOT NULL,
    training_requirements TEXT,
    experience_requirements TEXT,
    eligibility_requirements TEXT,
    special_requirements TEXT, -- Additional qualifications
    
    -- Application Details
    application_deadline DATE,
    application_instructions TEXT,
    required_documents TEXT, -- JSON array of required documents
    contact_email VARCHAR(100),
    contact_address TEXT,
    
    -- Template and Design
    color_scheme VARCHAR(20) DEFAULT 'blue', -- "blue", "teal", etc.
    banner_text VARCHAR(50) DEFAULT 'WE ARE HIRING',
    
    -- Status and Metadata
    status VARCHAR(20) DEFAULT 'draft', -- 'draft', 'published', 'closed', 'archived'
    created_by INTEGER, -- User who created this posting
    approved_by INTEGER, -- User who approved for publishing
    published_at TIMESTAMP NULL,
    closes_at TIMESTAMP NULL,
    
    -- Tracking
    view_count INTEGER DEFAULT 0,
    application_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Keys
    FOREIGN KEY (position_type_id) REFERENCES position_types(id),
    FOREIGN KEY (campus_id) REFERENCES campus_locations(id),
    FOREIGN KEY (created_by) REFERENCES users(id),
    FOREIGN KEY (approved_by) REFERENCES users(id)
);

-- Required documents template
CREATE TABLE IF NOT EXISTS required_documents_template (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name VARCHAR(200) NOT NULL,
    document_description TEXT,
    is_mandatory BOOLEAN DEFAULT TRUE,
    position_type_id INTEGER, -- If specific to position type
    display_order INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (position_type_id) REFERENCES position_types(id)
);

-- Job posting applications (links candidates to specific job postings)
CREATE TABLE IF NOT EXISTS job_applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_posting_id INTEGER NOT NULL,
    candidate_id INTEGER NOT NULL,
    application_status VARCHAR(20) DEFAULT 'submitted', -- 'submitted', 'under_review', 'shortlisted', 'rejected', 'hired'
    assessment_score DECIMAL(5,2), -- University assessment score
    assessment_breakdown TEXT, -- JSON of detailed scoring
    hr_notes TEXT,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP NULL,
    
    FOREIGN KEY (job_posting_id) REFERENCES lspu_job_postings(id),
    FOREIGN KEY (candidate_id) REFERENCES candidates(id),
    
    UNIQUE(job_posting_id, candidate_id) -- Prevent duplicate applications
);

-- Assessment criteria mapping for each job posting
CREATE TABLE IF NOT EXISTS job_assessment_criteria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_posting_id INTEGER NOT NULL,
    criteria_name VARCHAR(100) NOT NULL,
    criteria_weight DECIMAL(4,3), -- e.g., 0.25 for 25%
    min_score INTEGER DEFAULT 0,
    max_score INTEGER DEFAULT 100,
    description TEXT,
    
    FOREIGN KEY (job_posting_id) REFERENCES lspu_job_postings(id)
);

-- Insert default campus locations
INSERT OR IGNORE INTO campus_locations (campus_name, campus_code, contact_email) VALUES 
('LSPU - Santa Cruz Campus', 'SC', 'hrscc.recruitment@lspu.edu.ph'),
('LSPU - San Pablo City Campus', 'SPC', 'careers_spc@lspu.edu.ph'),
('LSPU - Los Baños Campus', 'LB', 'lspulbc.hrmo@lspu.edu.ph'),
('LSPU - Main Campus', 'MAIN', 'information.office@lspu.edu.ph');

-- Insert common required documents
INSERT OR IGNORE INTO required_documents_template (document_name, document_description, display_order) VALUES 
('Personal Data Sheet (PDS)', 'Fully accomplished Personal Data Sheet (PDS) with recent passport-sized picture (CSC Form No. 212, Rev. 2025); digitally signed or electronically signed', 1),
('Performance Rating', 'Performance rating in the last rating period (if applicable)', 2),
('Curriculum Vitae', 'Current curriculum vitae', 3),
('Certificate of Eligibility/Rating/License', 'Photocopy of certificate of eligibility/rating/license', 4),
('Transcript of Records', 'Photocopy of transcript of records', 5),
('Application Letter', 'Application letter addressed to the University President', 6);

-- Insert default university configuration
INSERT OR IGNORE INTO university_config (id) VALUES (1);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_lspu_job_postings_status ON lspu_job_postings(status);
CREATE INDEX IF NOT EXISTS idx_lspu_job_postings_position_type ON lspu_job_postings(position_type_id);
CREATE INDEX IF NOT EXISTS idx_lspu_job_postings_campus ON lspu_job_postings(campus_id);
CREATE INDEX IF NOT EXISTS idx_lspu_job_postings_deadline ON lspu_job_postings(application_deadline);
CREATE INDEX IF NOT EXISTS idx_job_applications_status ON job_applications(application_status);
CREATE INDEX IF NOT EXISTS idx_job_applications_score ON job_applications(assessment_score);