#!/usr/bin/env python3
"""
Debug the processing pipeline to find where the error occurs
"""

import sys
import os
import json
from utils import PersonalDataSheetProcessor

def debug_processing_pipeline():
    """Debug the complete processing pipeline"""
    print("🔍 Debugging processing pipeline...")
    
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
        print("\n1. Testing PersonalDataSheetProcessor initialization...")
        pds_processor = PersonalDataSheetProcessor()
        print("✅ PersonalDataSheetProcessor created")
        
        print("\n2. Testing process_excel_pds_file method...")
        result = pds_processor.process_excel_pds_file(test_file, test_file, mock_job)
        
        if result:
            print("✅ process_excel_pds_file returned data")
            print(f"📊 Result type: {type(result)}")
            print(f"📝 Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            # Check required fields for create_candidate
            required_fields = ['name', 'email', 'job_id', 'category', 'processing_type']
            missing_fields = []
            
            if isinstance(result, dict):
                for field in required_fields:
                    if field not in result:
                        missing_fields.append(field)
                        print(f"⚠️ Missing field: {field}")
                    else:
                        print(f"✅ Has field: {field} = {result[field]}")
            
            if missing_fields:
                print(f"❌ Missing required fields: {missing_fields}")
                return False
            else:
                print("✅ All required fields present")
                return True
        else:
            print("❌ process_excel_pds_file returned None")
            return False
            
    except Exception as e:
        print(f"❌ Error in processing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_processing_pipeline()
    print(f"\n{'=' * 50}")
    if success:
        print("🎉 Processing pipeline debug PASSED!")
    else:
        print("❌ Processing pipeline debug FAILED!")
    print(f"{'=' * 50}")