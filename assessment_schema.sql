-- University Assessment System Database Schema
-- This schema adds tables for the new assessment system

-- Position types configuration (Part-time Teaching, Regular Faculty, etc.)
CREATE TABLE IF NOT EXISTS position_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Assessment criteria templates for each position type
CREATE TABLE IF NOT EXISTS assessment_templates (
    id SERIAL PRIMARY KEY,
    position_type_id INTEGER REFERENCES position_types(id) ON DELETE CASCADE,
    criteria_category VARCHAR(50) NOT NULL, -- 'education', 'experience', 'training', etc.
    criteria_name VARCHAR(100) NOT NULL,
    max_points FLOAT NOT NULL,
    weight_percentage FLOAT NOT NULL,
    scoring_rules JSONB, -- Detailed scoring rules and thresholds
    is_automated BOOLEAN DEFAULT TRUE, -- Whether this can be scored automatically
    display_order INTEGER DEFAULT 0,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(position_type_id, criteria_category, criteria_name)
);

-- Position requirements and qualifications
CREATE TABLE IF NOT EXISTS position_requirements (
    id SERIAL PRIMARY KEY,
    job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    position_type_id INTEGER REFERENCES position_types(id) ON DELETE RESTRICT,
    minimum_education VARCHAR(100), -- 'Master', 'Doctoral', etc.
    required_experience INTEGER DEFAULT 0, -- minimum years
    required_certifications TEXT[], -- array of required certifications
    preferred_qualifications TEXT,
    subject_area VARCHAR(100), -- for teaching positions
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual candidate assessments
CREATE TABLE IF NOT EXISTS candidate_assessments (
    id SERIAL PRIMARY KEY,
    candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
    job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    position_type_id INTEGER REFERENCES position_types(id) ON DELETE RESTRICT,
    
    -- Automated scores (70% of total)
    education_score FLOAT DEFAULT 0,
    experience_score FLOAT DEFAULT 0,
    training_score FLOAT DEFAULT 0,
    eligibility_score FLOAT DEFAULT 0,
    accomplishments_score FLOAT DEFAULT 0,
    automated_total FLOAT DEFAULT 0,
    
    -- Manual scores (30% of total) - to be entered by HR
    interview_score FLOAT DEFAULT NULL,
    aptitude_score FLOAT DEFAULT NULL,
    manual_total FLOAT DEFAULT 0,
    
    -- Final assessment
    final_score FLOAT DEFAULT 0,
    rank_position INTEGER DEFAULT NULL,
    assessment_status VARCHAR(20) DEFAULT 'incomplete', -- 'incomplete', 'pending_interview', 'complete'
    recommendation VARCHAR(20) DEFAULT 'pending', -- 'highly_recommended', 'recommended', 'not_recommended'
    
    -- Metadata
    assessed_by INTEGER REFERENCES users(id),
    assessment_notes TEXT,
    score_breakdown JSONB, -- Detailed breakdown of all scores
    assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_date TIMESTAMP DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(candidate_id, job_id)
);

-- Manual assessment scores (interview components, aptitude test)
CREATE TABLE IF NOT EXISTS manual_assessment_scores (
    id SERIAL PRIMARY KEY,
    candidate_assessment_id INTEGER REFERENCES candidate_assessments(id) ON DELETE CASCADE,
    score_type VARCHAR(20) NOT NULL, -- 'interview' or 'aptitude'
    component_name VARCHAR(100) NOT NULL, -- 'personality', 'communication', etc.
    rating INTEGER, -- 1-10 scale for interview components, 1-5 for aptitude
    score FLOAT, -- calculated score based on rating
    max_possible FLOAT,
    notes TEXT,
    entered_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Assessment training data (from Excel files)
CREATE TABLE IF NOT EXISTS assessment_training_data (
    id SERIAL PRIMARY KEY,
    position_type_id INTEGER REFERENCES position_types(id) ON DELETE CASCADE,
    anonymized_data JSONB NOT NULL, -- PDS data with names removed
    scores JSONB NOT NULL, -- All assessment scores
    outcome VARCHAR(20), -- 'hired', 'not_hired', 'withdrawn'
    assessment_year INTEGER,
    data_source VARCHAR(50) DEFAULT 'excel_import',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Assessment comparison results (for ranking and analytics)
CREATE TABLE IF NOT EXISTS assessment_comparisons (
    id SERIAL PRIMARY KEY,
    job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    comparison_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    candidate_rankings JSONB NOT NULL, -- Array of candidate rankings with scores
    assessment_summary JSONB, -- Summary statistics
    generated_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default position types
INSERT INTO position_types (name, description) VALUES
('Part-time Teaching', 'Part-time instructional positions'),
('Regular Faculty', 'Full-time faculty positions with tenure track'),
('Non-Teaching Personnel', 'Administrative and support staff positions'),
('Job Order', 'Contractual and temporary positions')
ON CONFLICT (name) DO NOTHING;

-- Insert assessment criteria template for Part-time Teaching
INSERT INTO assessment_templates (position_type_id, criteria_category, criteria_name, max_points, weight_percentage, scoring_rules, is_automated, display_order, description) 
SELECT 
    pt.id,
    'potential',
    'Interview Score',
    70,
    10,
    '{"components": ["personality", "communication", "analytical", "achievement", "leadership", "relationship", "job_fit"], "scoring_scale": "1-10"}',
    false,
    1,
    'Interview assessment with 7 components'
FROM position_types pt WHERE pt.name = 'Part-time Teaching'
ON CONFLICT (position_type_id, criteria_category, criteria_name) DO NOTHING;

INSERT INTO assessment_templates (position_type_id, criteria_category, criteria_name, max_points, weight_percentage, scoring_rules, is_automated, display_order, description) 
SELECT 
    pt.id,
    'potential',
    'Aptitude Test',
    5,
    5,
    '{"scale": "1-5", "levels": {"5": "Superior", "4": "Above Average", "3": "Average", "2": "Below Average", "1": "Lowest"}}',
    false,
    2,
    'Aptitude test assessment'
FROM position_types pt WHERE pt.name = 'Part-time Teaching'
ON CONFLICT (position_type_id, criteria_category, criteria_name) DO NOTHING;

INSERT INTO assessment_templates (position_type_id, criteria_category, criteria_name, max_points, weight_percentage, scoring_rules, is_automated, display_order, description) 
SELECT 
    pt.id,
    'education',
    'Relevance and Appropriateness',
    40,
    32,
    '{"assessment_method": "degree_matching", "subject_relevance": true, "institution_quality": true}',
    true,
    3,
    'Relevance and appropriateness of educational background'
FROM position_types pt WHERE pt.name = 'Part-time Teaching'
ON CONFLICT (position_type_id, criteria_category, criteria_name) DO NOTHING;

INSERT INTO assessment_templates (position_type_id, criteria_category, criteria_name, max_points, weight_percentage, scoring_rules, is_automated, display_order, description) 
SELECT 
    pt.id,
    'education',
    'Basic Minimum Requirement',
    35,
    28,
    '{"minimum_degree": "Master", "required": true}',
    true,
    4,
    'Basic minimum educational requirement (Master''s degree)'
FROM position_types pt WHERE pt.name = 'Part-time Teaching'
ON CONFLICT (position_type_id, criteria_category, criteria_name) DO NOTHING;

INSERT INTO assessment_templates (position_type_id, criteria_category, criteria_name, max_points, weight_percentage, scoring_rules, is_automated, display_order, description) 
SELECT 
    pt.id,
    'education',
    'Doctoral Progress Bonus',
    5,
    4,
    '{"25_percent": 1, "50_percent": 2, "75_percent": 3, "CAR_complete": 4, "PhD_complete": 5}',
    true,
    5,
    'Additional points for doctoral degree progress'
FROM position_types pt WHERE pt.name = 'Part-time Teaching'
ON CONFLICT (position_type_id, criteria_category, criteria_name) DO NOTHING;

INSERT INTO assessment_templates (position_type_id, criteria_category, criteria_name, max_points, weight_percentage, scoring_rules, is_automated, display_order, description) 
SELECT 
    pt.id,
    'experience',
    'Years of Experience',
    20,
    20,
    '{"tiers": {"1-2_years": 5, "3-4_years": 10, "5-10_years": 15}, "bonus_per_year_over_10": 1, "relevance_multiplier": true}',
    true,
    6,
    'Years of relevant professional experience'
FROM position_types pt WHERE pt.name = 'Part-time Teaching'
ON CONFLICT (position_type_id, criteria_category, criteria_name) DO NOTHING;

INSERT INTO assessment_templates (position_type_id, criteria_category, criteria_name, max_points, weight_percentage, scoring_rules, is_automated, display_order, description) 
SELECT 
    pt.id,
    'training',
    'Professional Training',
    10,
    10,
    '{"baseline_hours": 40, "baseline_points": 5, "additional_per_8_hours": 1, "relevance_assessment": true}',
    true,
    7,
    'Professional training and development hours'
FROM position_types pt WHERE pt.name = 'Part-time Teaching'
ON CONFLICT (position_type_id, criteria_category, criteria_name) DO NOTHING;

INSERT INTO assessment_templates (position_type_id, criteria_category, criteria_name, max_points, weight_percentage, scoring_rules, is_automated, display_order, description) 
SELECT 
    pt.id,
    'eligibility',
    'Professional Eligibility',
    10,
    10,
    '{"certifications": ["RA 1080", "CSC Exam", "BAR/BOARD Exam"], "full_points_any": true}',
    true,
    8,
    'Professional licenses and eligibility certifications'
FROM position_types pt WHERE pt.name = 'Part-time Teaching'
ON CONFLICT (position_type_id, criteria_category, criteria_name) DO NOTHING;

INSERT INTO assessment_templates (position_type_id, criteria_category, criteria_name, max_points, weight_percentage, scoring_rules, is_automated, display_order, description) 
SELECT 
    pt.id,
    'accomplishments',
    'Outstanding Accomplishments',
    5,
    5,
    '{"types": ["citations", "recognitions", "honor_graduate", "board_topnotcher", "csc_topnotcher"], "points_per_accomplishment": 1}',
    true,
    9,
    'Awards, recognitions, and outstanding accomplishments'
FROM position_types pt WHERE pt.name = 'Part-time Teaching'
ON CONFLICT (position_type_id, criteria_category, criteria_name) DO NOTHING;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_assessment_templates_position_type ON assessment_templates(position_type_id);
CREATE INDEX IF NOT EXISTS idx_position_requirements_job_id ON position_requirements(job_id);
CREATE INDEX IF NOT EXISTS idx_candidate_assessments_candidate_id ON candidate_assessments(candidate_id);
CREATE INDEX IF NOT EXISTS idx_candidate_assessments_job_id ON candidate_assessments(job_id);
CREATE INDEX IF NOT EXISTS idx_candidate_assessments_status ON candidate_assessments(assessment_status);
CREATE INDEX IF NOT EXISTS idx_candidate_assessments_score ON candidate_assessments(final_score);
CREATE INDEX IF NOT EXISTS idx_manual_scores_assessment_id ON manual_assessment_scores(candidate_assessment_id);
CREATE INDEX IF NOT EXISTS idx_training_data_position_type ON assessment_training_data(position_type_id);
CREATE INDEX IF NOT EXISTS idx_assessment_comparisons_job_id ON assessment_comparisons(job_id);
