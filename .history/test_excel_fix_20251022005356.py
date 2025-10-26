#!/usr/bin/env python3
"""
Quick test to verify Excel PDS processing fixes
"""

import os
import sys

def test_excel_processing():
    """Test Excel PDS processing with the new method"""
    print("🧪 TESTING EXCEL PDS PROCESSING FIX")
    print("=" * 50)
    
    try:
        from utils import PersonalDataSheetProcessor, convert_pds_to_assessment_format
        
        # Initialize processor
        pds_processor = PersonalDataSheetProcessor()
        
        # Check if the extract_pds_data method exists
        if hasattr(pds_processor, 'extract_pds_data'):
            print("✅ extract_pds_data method found")
        else:
            print("❌ extract_pds_data method missing")
            return False
            
        # Check if the process_excel_pds_file method exists
        if hasattr(pds_processor, 'process_excel_pds_file'):
            print("✅ process_excel_pds_file method found")
        else:
            print("❌ process_excel_pds_file method missing")
            return False
        
        # Check if Excel files exist
        test_files = ["Sample PDS Lenar.xlsx", "Sample PDS New.xlsx", "sample_pds.xlsx"]
        found_files = [f for f in test_files if os.path.exists(f)]
        
        if not found_files:
            print("❌ No Excel PDS files found for testing")
            return False
        
        test_file = found_files[0]
        print(f"📄 Testing with: {test_file}")
        
        # Test the new extraction method
        print("\n🔍 Testing extract_pds_data method...")
        extracted_data = pds_processor.extract_pds_data(test_file)
        
        if extracted_data:
            print(f"✅ Data extracted successfully!")
            print(f"   Sections found: {list(extracted_data.keys())}")
            
            # Test conversion
            print("\n🔄 Testing conversion to assessment format...")
            converted_data = convert_pds_to_assessment_format(extracted_data, test_file)
            
            if converted_data:
                print(f"✅ Conversion successful!")
                print(f"   Education entries: {len(converted_data.get('education', []))}")
                print(f"   Experience entries: {len(converted_data.get('experience', []))}")
                print(f"   Training entries: {len(converted_data.get('training', []))}")
                
                # Test the complete method
                print("\n🎯 Testing complete process_excel_pds_file method...")
                result = pds_processor.process_excel_pds_file(test_file, test_file, None)
                
                if result:
                    print(f"✅ Complete processing successful!")
                    print(f"   Score: {result.get('score', 'N/A')}")
                    print(f"   Type: {result.get('processing_type', 'N/A')}")
                    return True
                else:
                    print("❌ Complete processing failed")
                    return False
            else:
                print("❌ Conversion failed")
                return False
        else:
            print("❌ No data extracted")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    success = test_excel_processing()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Excel PDS processing fix verified!")
        print("\n📋 Next step: Update app.py to use process_excel_pds_file method")
    else:
        print("❌ TESTS FAILED!")
        print("🔧 Additional fixes needed")

if __name__ == "__main__":
    main()