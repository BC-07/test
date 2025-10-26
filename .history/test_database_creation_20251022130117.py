#!/usr/bin/env python3
"""
Test the database create_candidate method with the actual data structure
"""

import sys
import os
import json
from utils import PersonalDataSheetProcessor
from database import DatabaseManager

def test_database_creation():
    """Test the database create_candidate method"""
    print("🔍 Testing database candidate creation...")
    
    # Test file
    test_file = "Sample PDS New.xlsx"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    # Mock job object
    mock_job = {
        'id': 1,
        'title': 'Test Job',
        'source': 'LSPU'
    }
    
    try:
        print("\n1. Getting candidate data from processing pipeline...")
        pds_processor = PersonalDataSheetProcessor()
        candidate_data = pds_processor.process_excel_pds_file(test_file, test_file, mock_job)
        
        if not candidate_data:
            print("❌ No candidate data returned")
            return False
        
        print("✅ Candidate data generated successfully")
        print(f"📝 Name: {candidate_data.get('name')}")
        print(f"📧 Email: {candidate_data.get('email')}")
        print(f"🆔 Job ID: {candidate_data.get('job_id')}")
        
        print("\n2. Testing database connection...")
        db_manager = DatabaseManager()
        print("✅ Database manager created")
        
        print("\n3. Testing create_candidate method...")
        try:
            candidate_id = db_manager.create_candidate(candidate_data)
            
            if candidate_id:
                print(f"✅ Successfully created candidate with ID: {candidate_id}")
                print("✅ Database creation works perfectly!")
                return True
            else:
                print("❌ create_candidate returned None")
                return False
                
        except Exception as db_error:
            print(f"❌ Database error: {db_error}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_database_creation()
    print(f"\n{'=' * 50}")
    if success:
        print("🎉 Database creation test PASSED!")
        print("✅ The issue is NOT in the database creation")
    else:
        print("❌ Database creation test FAILED!")
        print("❌ The issue IS in the database creation")
    print(f"{'=' * 50}")