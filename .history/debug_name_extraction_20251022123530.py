#!/usr/bin/env python3
"""
Debug the specific name extraction issue
"""

def debug_name_extraction():
    """Debug why name extraction is showing Unknown Candidate"""
    print("🔍 DEBUGGING NAME EXTRACTION")
    print("=" * 50)
    
    try:
        # Test direct ImprovedPDSExtractor
        from improved_pds_extractor import ImprovedPDSExtractor
        extractor = ImprovedPDSExtractor()
        
        test_file = "Sample PDS New.xlsx"
        extracted_data = extractor.extract_pds_data(test_file)
        
        print("📊 Raw ImprovedPDSExtractor Output:")
        if 'personal_info' in extracted_data:
            personal_info = extracted_data['personal_info']
            print(f"   First Name: {personal_info.get('first_name', 'MISSING')}")
            print(f"   Middle Name: {personal_info.get('middle_name', 'MISSING')}")
            print(f"   Surname: {personal_info.get('surname', 'MISSING')}")
            print(f"   Email: {personal_info.get('email', 'MISSING')}")
            print(f"   Mobile: {personal_info.get('mobile_no', 'MISSING')}")
            
            # Test name construction
            name = f"{personal_info.get('first_name', '')} {personal_info.get('middle_name', '')} {personal_info.get('surname', '')}".strip()
            print(f"   🔤 Constructed Name: '{name}'")
        else:
            print("   ❌ No personal_info section found!")
            
        # Test conversion
        print("\n🔄 Testing Conversion:")
        from improved_pds_converter import convert_improved_pds_to_assessment_format
        converted_data = convert_improved_pds_to_assessment_format(extracted_data)
        
        basic_info = converted_data.get('basic_info', {})
        print(f"   Converted Name: '{basic_info.get('name', 'MISSING')}'")
        print(f"   Converted Email: '{basic_info.get('email', 'MISSING')}'")
        print(f"   Converted Phone: '{basic_info.get('phone', 'MISSING')}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_name_extraction()