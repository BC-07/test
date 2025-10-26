#!/usr/bin/env python3
"""
Test the exact Flask app processing chain to find the issue
"""

import sys
import os
import json
import tempfile
import shutil
from utils import PersonalDataSheetProcessor

def test_flask_processing_chain():
    """Test the exact Flask app processing chain"""
    print("🔍 Testing Flask app processing chain...")
    
    # Test file
    test_file = "Sample PDS New.xlsx"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    # Create temp file like Flask does
    temp_dir = "temp_uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    temp_filename = f"test_Sample_PDS_New.xlsx"
    temp_path = os.path.join(temp_dir, temp_filename)
    shutil.copy2(test_file, temp_path)
    
    # Mock job object like Flask uses
    mock_job = {
        'id': 2,  # Using job ID 2 like in the logs
        'title': 'Instructor I',
        'source': 'LSPU'
    }
    
    # Mock file record like Flask uses
    file_record = {
        'temp_path': temp_path,
        'original_name': 'Sample PDS New.xlsx',
        'file_type': 'excel'
    }
    
    try:
        print("\n1. Testing _detect_pds_file logic...")
        # Simulate _detect_pds_file
        is_pds = file_record['file_type'] == 'excel'  # Simplified
        print(f"✅ Detected as PDS: {is_pds}")
        
        print("\n2. Testing _process_pds_file logic...")
        if is_pds and file_record['file_type'] == 'excel':
            print("📋 Processing as PDS (Excel)")
            
            print("\n3. Testing _process_excel_file logic...")
            # Create PDS processor like Flask does
            pds_processor = PersonalDataSheetProcessor()
            
            if pds_processor:
                print("✅ PDS processor available")
                
                print("\n4. Testing process_excel_pds_file call...")
                result = pds_processor.process_excel_pds_file(
                    file_record['temp_path'], 
                    file_record['original_name'], 
                    mock_job
                )
                
                if result:
                    print("✅ process_excel_pds_file returned data")
                    print(f"📊 Result type: {type(result)}")
                    print(f"📝 Name: {result.get('name')}")
                    print(f"📧 Email: {result.get('email')}")
                    print(f"🆔 Job ID: {result.get('job_id')}")
                    
                    # Check if this matches what create_candidate expects
                    required_fields = ['name', 'email', 'job_id', 'category', 'processing_type']
                    all_present = all(field in result for field in required_fields)
                    print(f"✅ All required fields present: {all_present}")
                    
                    return True
                else:
                    print("❌ process_excel_pds_file returned None")
                    return False
            else:
                print("❌ PDS processor not available")
                return False
        else:
            print("❌ Not detected as Excel PDS")
            return False
            
    except Exception as e:
        print(f"❌ Error in Flask processing chain: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"🗑️ Cleaned up temp file: {temp_path}")

if __name__ == "__main__":
    success = test_flask_processing_chain()
    print(f"\n{'=' * 50}")
    if success:
        print("🎉 Flask processing chain test PASSED!")
        print("✅ The issue is NOT in the processing chain")
    else:
        print("❌ Flask processing chain test FAILED!")
        print("❌ Found the issue in the processing chain")
    print(f"{'=' * 50}")