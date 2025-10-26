#!/usr/bin/env python3
"""
Debug test for ImprovedPDSExtractor integration
Check if the integration is working properly
"""

import os
import sys

def test_improved_extractor_directly():
    """Test ImprovedPDSExtractor directly"""
    print("🔍 TESTING ImprovedPDSExtractor DIRECTLY")
    print("=" * 50)
    
    try:
        from improved_pds_extractor import ImprovedPDSExtractor
        extractor = ImprovedPDSExtractor()
        
        test_file = "Sample PDS New.xlsx"
        if not os.path.exists(test_file):
            print(f"❌ Test file {test_file} not found")
            return False
            
        print(f"📄 Testing extraction from: {test_file}")
        extracted_data = extractor.extract_pds_data(test_file)
        
        if extracted_data:
            print("✅ ImprovedPDSExtractor working correctly!")
            print(f"📊 Sections extracted: {list(extracted_data.keys())}")
            
            # Show some details
            if 'personal_info' in extracted_data:
                personal = extracted_data['personal_info']
                print(f"👤 Name: {personal.get('first_name', '')} {personal.get('surname', '')}")
                print(f"📧 Email: {personal.get('email', 'N/A')}")
            
            if 'educational_background' in extracted_data:
                print(f"🎓 Education entries: {len(extracted_data['educational_background'])}")
                
            if 'work_experience' in extracted_data:
                print(f"💼 Work entries: {len(extracted_data['work_experience'])}")
                
            return True
        else:
            print("❌ ImprovedPDSExtractor returned no data")
            return False
            
    except Exception as e:
        print(f"❌ Error with ImprovedPDSExtractor: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utils_integration():
    """Test the utils.py integration"""
    print("\n🔍 TESTING UTILS.PY INTEGRATION")
    print("=" * 50)
    
    try:
        from utils import PersonalDataSheetProcessor
        processor = PersonalDataSheetProcessor()
        
        test_file = "Sample PDS New.xlsx"
        if not os.path.exists(test_file):
            print(f"❌ Test file {test_file} not found")
            return False
            
        print(f"📄 Testing extraction via utils.py: {test_file}")
        extracted_data = processor.extract_pds_data(test_file)
        
        if extracted_data:
            print("✅ Utils.py extraction working!")
            print(f"📊 Sections extracted: {list(extracted_data.keys())}")
            return True
        else:
            print("❌ Utils.py extraction returned no data")
            return False
            
    except Exception as e:
        print(f"❌ Error with utils.py integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conversion_process():
    """Test the full conversion process"""
    print("\n🔍 TESTING CONVERSION PROCESS")
    print("=" * 50)
    
    try:
        from utils import PersonalDataSheetProcessor, convert_pds_to_assessment_format
        processor = PersonalDataSheetProcessor()
        
        test_file = "Sample PDS New.xlsx"
        print(f"📄 Testing full process: {test_file}")
        
        # Extract data
        extracted_data = processor.extract_pds_data(test_file)
        print(f"📊 Extracted sections: {list(extracted_data.keys())}")
        
        # Convert to assessment format
        converted_data = convert_pds_to_assessment_format(extracted_data, test_file)
        print(f"🔄 Converted sections: {list(converted_data.keys())}")
        
        # Check basic info
        basic_info = converted_data.get('basic_info', {})
        print(f"👤 Converted name: {basic_info.get('name', 'N/A')}")
        print(f"📧 Converted email: {basic_info.get('email', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in conversion process: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests"""
    print("🧪 DEBUGGING IMPROVED PDS EXTRACTOR INTEGRATION")
    print("=" * 60)
    
    # Test 1: Direct extractor
    test1_result = test_improved_extractor_directly()
    
    # Test 2: Utils integration
    test2_result = test_utils_integration()
    
    # Test 3: Conversion process
    test3_result = test_conversion_process()
    
    print("\n" + "=" * 60)
    print("🏁 DEBUG TEST SUMMARY")
    print("=" * 60)
    print(f"Direct ImprovedPDSExtractor: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"Utils.py Integration: {'✅ PASS' if test2_result else '❌ FAIL'}")
    print(f"Conversion Process: {'✅ PASS' if test3_result else '❌ FAIL'}")
    
    if all([test1_result, test2_result, test3_result]):
        print("\n🎉 ALL TESTS PASSED! Integration should be working.")
    else:
        print("\n❌ SOME TESTS FAILED! Need to fix integration issues.")

if __name__ == "__main__":
    main()