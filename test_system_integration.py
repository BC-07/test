"""
Test script to verify the system is using the enhanced extractor
"""

import os
import sys

def test_system_integration():
    print("=== SYSTEM INTEGRATION TEST ===")
    print("Testing if the main application uses our enhanced ImprovedPDSExtractor\n")
    
    # Test 1: Direct ImprovedPDSExtractor test
    print("1. Testing ImprovedPDSExtractor directly...")
    try:
        from improved_pds_extractor import ImprovedPDSExtractor
        
        # Test PDF file
        pdf_file = "Sample PDS Page 1.pdf"
        if os.path.exists(pdf_file):
            extractor = ImprovedPDSExtractor()
            result = extractor.extract_pds_data(pdf_file)
            
            print(f"✅ Direct extraction successful")
            print(f"Personal info fields: {len(result.get('personal_info', {}))}")
            print(f"Other information fields: {len(result.get('other_information', {}))}")
            
            # Check for our enhanced fields
            other_info = result.get('other_information', {})
            enhanced_fields = ['related_by_consanguinity', 'administrative_offense', 'criminally_charged']
            found_enhanced = sum(1 for field in enhanced_fields if field in other_info)
            
            print(f"Enhanced fields found: {found_enhanced}/{len(enhanced_fields)}")
            if found_enhanced > 0:
                print("✅ Enhanced extraction patterns working")
            else:
                print("❌ Enhanced extraction patterns NOT working")
                
        else:
            print(f"❌ PDF file {pdf_file} not found")
            
    except Exception as e:
        print(f"❌ Direct extractor test failed: {e}")
    
    print("\n" + "="*50)
    
    # Test 2: PersonalDataSheetProcessor integration
    print("2. Testing PersonalDataSheetProcessor integration...")
    try:
        from utils import PersonalDataSheetProcessor
        
        processor = PersonalDataSheetProcessor()
        
        # Test with PDF
        pdf_file = "Sample PDS Page 1.pdf"
        if os.path.exists(pdf_file):
            result = processor.extract_pds_data(pdf_file)
            
            if result:
                print(f"✅ PersonalDataSheetProcessor extraction successful")
                print(f"Personal info fields: {len(result.get('personal_info', {}))}")
                print(f"Other information fields: {len(result.get('other_information', {}))}")
                
                # Check for our enhanced fields
                other_info = result.get('other_information', {})
                enhanced_fields = ['related_by_consanguinity', 'administrative_offense', 'criminally_charged']
                found_enhanced = sum(1 for field in enhanced_fields if field in other_info)
                
                print(f"Enhanced fields found: {found_enhanced}/{len(enhanced_fields)}")
                if found_enhanced > 0:
                    print("✅ PersonalDataSheetProcessor using enhanced extraction")
                else:
                    print("❌ PersonalDataSheetProcessor NOT using enhanced extraction")
                    
                # Show what's actually in other_information
                print("\nActual other_information content:")
                for key, value in other_info.items():
                    print(f"  {key}: {value}")
                    
            else:
                print("❌ PersonalDataSheetProcessor returned None")
        else:
            print(f"❌ PDF file {pdf_file} not found")
            
    except Exception as e:
        print(f"❌ PersonalDataSheetProcessor test failed: {e}")
    
    print("\n" + "="*50)
    
    # Test 3: Check if there are multiple extractor files
    print("3. Checking for multiple extractor files...")
    
    extractor_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if 'extractor' in file.lower() and file.endswith('.py'):
                extractor_files.append(os.path.join(root, file))
    
    print("Found extractor files:")
    for file in extractor_files:
        print(f"  {file}")
        
    # Check imports in utils.py
    print("\n4. Checking utils.py imports...")
    try:
        with open('utils.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for extractor imports
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'import' in line and 'extractor' in line.lower():
                print(f"  Line {i+1}: {line.strip()}")
                
    except Exception as e:
        print(f"❌ Error checking utils.py: {e}")
    
    print("\n" + "="*50)
    print("Test complete!")

if __name__ == "__main__":
    test_system_integration()