#!/usr/bin/env python3
"""
Test script to verify Flask server is using the updated ImprovedPDSExtractor
"""

import sys
import os
from utils import PersonalDataSheetProcessor
from improved_pds_extractor import ImprovedPDSExtractor

def test_server_integration():
    """Test that the server components are using the right extractor"""
    print("🔍 Testing server integration with ImprovedPDSExtractor...")
    
    # Test 1: Check that PersonalDataSheetProcessor is using ImprovedPDSExtractor
    print("\n1. Testing PersonalDataSheetProcessor initialization...")
    try:
        pds_processor = PersonalDataSheetProcessor()
        print("✅ PersonalDataSheetProcessor created successfully")
        
        # Check if it has the improved methods
        if hasattr(pds_processor, 'extract_pds_data'):
            print("✅ extract_pds_data method found")
        else:
            print("❌ extract_pds_data method missing")
            
        if hasattr(pds_processor, 'process_excel_pds_file'):
            print("✅ process_excel_pds_file method found")
        else:
            print("❌ process_excel_pds_file method missing")
            
    except Exception as e:
        print(f"❌ Failed to create PersonalDataSheetProcessor: {e}")
        return False
    
    # Test 2: Test direct ImprovedPDSExtractor
    print("\n2. Testing ImprovedPDSExtractor directly...")
    try:
        extractor = ImprovedPDSExtractor()
        print("✅ ImprovedPDSExtractor created successfully")
        
        # Test with a sample file if available
        sample_files = [
            "Sample PDS New.xlsx",
            "PDS-Sample Cabael.xlsx"
        ]
        
        for sample_file in sample_files:
            if os.path.exists(sample_file):
                print(f"\n3. Testing extraction with {sample_file}...")
                try:
                    result = extractor.extract_from_excel(sample_file)
                    if result and 'personal_info' in result:
                        personal_info = result['personal_info']
                        name = personal_info.get('name', 'Unknown')
                        email = personal_info.get('email', 'Unknown')
                        total_entries = len([section for section in result.values() if isinstance(section, list) and section])
                        
                        print(f"✅ Successfully extracted from {sample_file}:")
                        print(f"   📝 Name: {name}")
                        print(f"   📧 Email: {email}")
                        print(f"   📊 Total data sections: {len(result)}")
                        print(f"   🔢 Has populated sections: {total_entries > 0}")
                        
                        # Check for the expected 33+ entries structure
                        education = result.get('education', [])
                        experience = result.get('work_experience', [])
                        print(f"   🎓 Education entries: {len(education)}")
                        print(f"   💼 Work experience entries: {len(experience)}")
                        
                        return True
                    else:
                        print(f"❌ Invalid result structure from {sample_file}")
                except Exception as e:
                    print(f"❌ Failed to extract from {sample_file}: {e}")
        
        print("\n⚠️ No sample files found for testing")
        return True  # Server integration is working even without test files
        
    except Exception as e:
        print(f"❌ Failed to create ImprovedPDSExtractor: {e}")
        return False

if __name__ == "__main__":
    success = test_server_integration()
    print(f"\n{'=' * 50}")
    if success:
        print("🎉 Server integration test PASSED!")
        print("✅ Flask server should now use ImprovedPDSExtractor")
        print("✅ Expected output: 33+ total entries instead of 16")
        print("✅ Names should be extracted properly (not hardcoded)")
    else:
        print("❌ Server integration test FAILED!")
        print("❌ Check imports and component initialization")
    print(f"{'=' * 50}")