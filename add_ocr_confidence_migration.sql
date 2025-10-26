-- Migration to add OCR confidence field to candidates table
-- Run this to add OCR confidence support

-- Add OCR confidence field to candidates table
ALTER TABLE candidates 
ADD COLUMN IF NOT EXISTS ocr_confidence FLOAT DEFAULT NULL;

-- Add comment for the new field
COMMENT ON COLUMN candidates.ocr_confidence IS 'OCR extraction confidence score (0-100) for scanned documents';

-- Update processing_type comment to include new OCR option
COMMENT ON COLUMN candidates.processing_type IS 'Type of processing used: resume, pds, pds_text, pds_only, ocr_scanned';

-- Create index on processing_type for better query performance
CREATE INDEX IF NOT EXISTS idx_candidates_processing_type ON candidates(processing_type);

-- Create index on ocr_confidence for filtering by confidence level
CREATE INDEX IF NOT EXISTS idx_candidates_ocr_confidence ON candidates(ocr_confidence) WHERE ocr_confidence IS NOT NULL;

-- Add check constraint to ensure confidence is between 0 and 100
ALTER TABLE candidates 
ADD CONSTRAINT IF NOT EXISTS chk_ocr_confidence_range 
CHECK (ocr_confidence IS NULL OR (ocr_confidence >= 0 AND ocr_confidence <= 100));

-- Update any existing OCR scanned records that might not have confidence
-- (This would be run after the application starts using the new field)
-- UPDATE candidates 
-- SET ocr_confidence = 0 
-- WHERE processing_type = 'ocr_scanned' AND ocr_confidence IS NULL;

COMMIT;