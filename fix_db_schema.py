#!/usr/bin/env python3
"""
Fix database schema issues
- Fix recommendation field size from VARCHAR(20) to VARCHAR(50)
"""

import os
import sys
from database import DatabaseManager

def fix_recommendation_field():
    """Fix the recommendation field size issue"""
    print("🔧 Fixing recommendation field size...")
    
    db_manager = DatabaseManager()
    
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check current column size
                cursor.execute("""
                    SELECT character_maximum_length 
                    FROM information_schema.columns 
                    WHERE table_name = 'assessments' 
                    AND column_name = 'recommendation'
                """)
                current_size = cursor.fetchone()
                
                if current_size and current_size[0] == 20:
                    print(f"📏 Current recommendation field size: {current_size[0]} characters")
                    print("🔄 Updating to 50 characters...")
                    
                    # Alter the column
                    cursor.execute("ALTER TABLE assessments ALTER COLUMN recommendation TYPE VARCHAR(50)")
                    
                    # Update the comment
                    cursor.execute("""
                        COMMENT ON COLUMN assessments.recommendation IS 
                        'Assessment recommendation: highly_recommended, recommended, conditionally_recommended, not_recommended, pending'
                    """)
                    
                    conn.commit()
                    print("✅ Recommendation field updated successfully!")
                    
                elif current_size and current_size[0] >= 50:
                    print(f"✅ Recommendation field already has sufficient size: {current_size[0]} characters")
                else:
                    print("❌ Could not determine current field size")
                    
    except Exception as e:
        print(f"❌ Error fixing recommendation field: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🛠️  Database Schema Fixer")
    print("=" * 40)
    
    if fix_recommendation_field():
        print("\n✅ All database fixes applied successfully!")
    else:
        print("\n❌ Some fixes failed - check the logs above")
        sys.exit(1)

if __name__ == "__main__":
    main()