-- Fix recommendation field size
-- Current: VARCHAR(20) is too small for 'conditionally_recommended' (23 chars)
-- Solution: Increase to VARCHAR(50) to accommodate all recommendation types

ALTER TABLE assessments ALTER COLUMN recommendation TYPE VARCHAR(50);

-- Update the comment to reflect the new allowed values
COMMENT ON COLUMN assessments.recommendation IS 'Assessment recommendation: highly_recommended, recommended, conditionally_recommended, not_recommended, pending';