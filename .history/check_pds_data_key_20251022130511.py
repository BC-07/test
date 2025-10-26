#!/usr/bin/env python3
"""
Check if the result from process_excel_pds_file has the pds_data key
"""

import sys
import os
from utils import PersonalDataSheetProcessor

def check_pds_data_key():
    """Check if the result has the required pds_data key"""
    print("🔍 Checking pds_data key in result...")
    
    test_file = "Sample PDS New.xlsx"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    mock_job = {
        'id': 2,
        'title': 'Instructor I',
        'source': 'LSPU'
    }
    
    try:
        pds_processor = PersonalDataSheetProcessor()
        result = pds_processor.process_excel_pds_file(test_file, test_file, mock_job)
        
        if result:
            print("✅ Got result from process_excel_pds_file")
            print(f"📊 Result type: {type(result)}")
            print(f"📝 Keys: {list(result.keys())}")
            
            # Check for pds_data key specifically
            if 'pds_data' in result:
                print("✅ 'pds_data' key found in result")
                pds_data = result['pds_data']
                print(f"📊 pds_data type: {type(pds_data)}")
                print(f"📝 pds_data keys: {list(pds_data.keys()) if isinstance(pds_data, dict) else 'Not a dict'}")
                return True
            else:
                print("❌ 'pds_data' key NOT found in result")
                print("🔍 This is why Flask processing fails!")
                return False
        else:
            print("❌ No result from process_excel_pds_file")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_pds_data_key()
    print(f"\n{'=' * 50}")
    if success:
        print("🎉 pds_data key check PASSED!")
    else:
        print("❌ pds_data key check FAILED!")
        print("🔧 Need to fix the pds_data key structure")
    print(f"{'=' * 50}")