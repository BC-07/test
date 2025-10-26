"""
URGENT: Comprehensive diagnostic test to identify all remaining PDF extraction issues
This will be our starting point tomorrow to systematically fix everything
"""

from improved_pds_extractor import ImprovedPDSExtractor
from utils import PersonalDataSheetProcessor
import json

def comprehensive_diagnostic():
    print("="*80)
    print("🚨 COMPREHENSIVE PDF EXTRACTION DIAGNOSTIC")
    print("="*80)
    
    print("\n1. TESTING RAW PDF EXTRACTION...")
    print("-" * 50)
    
    extractor = ImprovedPDSExtractor()
    pdf_result = extractor.extract_pds_data('SamplePDS_Merged.pdf')
    
    if pdf_result:
        print("✅ Raw PDF extraction successful")
        
        # Check each section in detail
        sections_to_check = {
            'personal_info': 'Personal Information',
            'educational_background': 'Education',
            'work_experience': 'Work Experience', 
            'civil_service_eligibility': 'Civil Service',
            'learning_development': 'Training/Learning',
            'voluntary_work': 'Voluntary Work',
            'other_information': 'Other Information'
        }
        
        for key, name in sections_to_check.items():
            data = pdf_result.get(key, {})
            if isinstance(data, list):
                print(f"  {name}: {len(data)} entries")
                if data and len(data) > 0:
                    print(f"    Sample: {str(data[0])[:100]}...")
                else:
                    print("    ❌ NO DATA EXTRACTED")
            elif isinstance(data, dict):
                print(f"  {name}: {len(data)} fields")
                if data:
                    sample_keys = list(data.keys())[:3]
                    print(f"    Keys: {sample_keys}...")
                else:
                    print("    ❌ NO DATA EXTRACTED")
    else:
        print("❌ Raw PDF extraction FAILED")
        return
    
    print("\n2. TESTING EXCEL EXTRACTION FOR COMPARISON...")
    print("-" * 50)
    
    try:
        xlsx_result = extractor.extract_pds_data('Sample PDS Lenar.xlsx')
        if xlsx_result:
            print("✅ Excel extraction successful")
            
            for key, name in sections_to_check.items():
                data = xlsx_result.get(key, {})
                if isinstance(data, list):
                    print(f"  {name}: {len(data)} entries")
                elif isinstance(data, dict):
                    print(f"  {name}: {len(data)} fields")
        else:
            print("❌ Excel extraction failed")
    except Exception as e:
        print(f"❌ Excel test error: {e}")
    
    print("\n3. TESTING SYSTEM INTEGRATION...")
    print("-" * 50)
    
    processor = PersonalDataSheetProcessor()
    
    # Test PDS detection
    is_pds = processor.is_pds_file('SamplePDS_Merged.pdf')
    print(f"PDS Detection: {'✅ TRUE' if is_pds else '❌ FALSE'}")
    
    if is_pds:
        # Test system extraction
        system_result = processor.extract_pds_data('SamplePDS_Merged.pdf')
        if system_result:
            print("✅ System extraction successful")
        else:
            print("❌ System extraction failed")
            
        # Test candidate creation
        candidate = processor.process_pds_candidate('SamplePDS_Merged.pdf')
        if candidate:
            print("✅ Candidate creation successful")
            
            # Check what assessment engine will receive
            print("\n4. ASSESSMENT ENGINE DATA ANALYSIS...")
            print("-" * 50)
            
            critical_fields = {
                'name': 'Full Name',
                'email': 'Email Address',
                'education': 'Education Records',
                'experience': 'Work Experience',
                'training': 'Training Records',
                'skills': 'Skills List',
                'other_information': 'Other Information'
            }
            
            for field, description in critical_fields.items():
                value = candidate.get(field)
                if value:
                    if isinstance(value, list):
                        print(f"  ✅ {description}: {len(value)} entries")
                        if len(value) == 0:
                            print(f"    ⚠️  WARNING: Empty list!")
                    elif isinstance(value, dict):
                        print(f"  ✅ {description}: {len(value)} fields")
                        if len(value) == 0:
                            print(f"    ⚠️  WARNING: Empty dict!")
                    else:
                        print(f"  ✅ {description}: {str(value)[:50]}...")
                else:
                    print(f"  ❌ {description}: MISSING")
            
            # Check file type metadata
            print(f"\nFile Type Information:")
            processing_type = candidate.get('processing_type', 'UNKNOWN')
            print(f"  Processing Type: {processing_type}")
            
            if 'pds_data' in candidate:
                metadata = candidate['pds_data'].get('extraction_metadata', {})
                file_type = metadata.get('file_type', 'UNKNOWN')
                print(f"  Metadata File Type: {file_type}")
                
                if file_type != 'PDF':
                    print(f"  ❌ ISSUE: PDF file incorrectly labeled as {file_type}")
                else:
                    print(f"  ✅ File type correctly identified as PDF")
        else:
            print("❌ Candidate creation failed")
    
    print("\n5. SPECIFIC ISSUE IDENTIFICATION...")
    print("-" * 50)
    
    issues_found = []
    
    # Check for specific known issues
    if pdf_result:
        # Education issue check
        edu_data = pdf_result.get('educational_background', [])
        if len(edu_data) == 0:
            issues_found.append("❌ No educational background extracted")
        else:
            # Check if school names are extracted properly
            for i, edu in enumerate(edu_data[:3]):
                school_name = edu.get('school_name', '')
                if not school_name or len(school_name) < 3:
                    issues_found.append(f"❌ Education entry {i+1} has invalid school name: '{school_name}'")
        
        # Work experience issue check
        work_data = pdf_result.get('work_experience', [])
        if len(work_data) == 0:
            issues_found.append("❌ No work experience extracted")
        
        # Other information issue check
        other_data = pdf_result.get('other_information', {})
        if len(other_data) == 0:
            issues_found.append("❌ No other information extracted")
        else:
            yes_no_count = sum(1 for v in other_data.values() if v in ['YES', 'NO'])
            if yes_no_count == 0:
                issues_found.append("❌ No yes/no questions found in other information")
    
    if issues_found:
        print("CRITICAL ISSUES IDENTIFIED:")
        for issue in issues_found:
            print(f"  {issue}")
    else:
        print("✅ No critical issues found - data extraction appears complete")
    
    print("\n" + "="*80)
    print("🎯 DIAGNOSTIC COMPLETE")
    print("="*80)
    
    if issues_found:
        print(f"\n📋 TOMORROW'S PRIORITIES:")
        print("1. Fix the critical issues listed above")
        print("2. Test with multiple PDF files") 
        print("3. Compare PDF vs Excel assessment scores")
        print("4. Verify UI display of all sections")
        print("5. Complete end-to-end workflow testing")
    else:
        print("\n🎉 SYSTEM APPEARS TO BE WORKING CORRECTLY!")
        print("Focus tomorrow on:")
        print("1. User interface improvements")
        print("2. Performance optimization") 
        print("3. Additional file format support")
        print("4. Assessment algorithm refinement")

if __name__ == "__main__":
    comprehensive_diagnostic()