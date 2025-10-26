#!/usr/bin/env python3
"""
Database Migration: Fix recommendation field size
- Increase recommendation field from VARCHAR(20) to VARCHAR(50)
"""

import sys
from database import DatabaseManager

def main():
    """Apply database migration"""
    print("🔧 Applying database migration: Fix recommendation field size")
    
    try:
        db_manager = DatabaseManager()
        
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check if assessments table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'assessments'
                    )
                """)
                
                if not cursor.fetchone()[0]:
                    print("ℹ️  Assessments table doesn't exist - creating it...")
                    
                    # Create assessments table with proper field sizes
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS assessments (
                            id SERIAL PRIMARY KEY,
                            candidate_id INTEGER REFERENCES candidates(id) ON DELETE CASCADE,
                            job_id INTEGER,
                            automated_score NUMERIC(5,2) DEFAULT 0,
                            percentage_score NUMERIC(5,2) DEFAULT 0,
                            assessment_results JSONB,
                            assessment_status VARCHAR(20) DEFAULT 'incomplete',
                            recommendation VARCHAR(50) DEFAULT 'pending',
                            assessment_notes TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    print("✅ Created assessments table with VARCHAR(50) recommendation field")
                    
                else:
                    # Check current field size
                    cursor.execute("""
                        SELECT character_maximum_length 
                        FROM information_schema.columns 
                        WHERE table_name = 'assessments' 
                        AND column_name = 'recommendation'
                    """)
                    
                    result = cursor.fetchone()
                    if result and result[0] < 50:
                        print(f"📏 Current recommendation field size: {result[0]} characters")
                        print("🔄 Updating to 50 characters...")
                        
                        cursor.execute("""
                            ALTER TABLE assessments 
                            ALTER COLUMN recommendation TYPE VARCHAR(50)
                        """)
                        
                        print("✅ Updated recommendation field to VARCHAR(50)")
                    else:
                        print("✅ Recommendation field already has sufficient size")
                
                conn.commit()
                print("✅ Migration completed successfully!")
                
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()