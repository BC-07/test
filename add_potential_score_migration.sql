-- Migration to add potential_score column to candidates table
-- This column stores the manual potential score (0-15 points) for university assessment

ALTER TABLE candidates ADD COLUMN IF NOT EXISTS potential_score FLOAT DEFAULT 0.0;

-- Add comment to document the field
COMMENT ON COLUMN candidates.potential_score IS 'Manual potential score for university assessment (0-15 points)';

-- Update any existing candidates to have default potential score of 0
UPDATE candidates SET potential_score = 0.0 WHERE potential_score IS NULL;