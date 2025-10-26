#!/usr/bin/env python3
"""
PDS Extractor Comparison Test
Compare the extraction capabilities of different PDS extractors using the same file
"""

import os
import json
from datetime import datetime
from typing import Dict, Any

def test_basic_pds_processor():
    """Test the current PersonalDataSheetProcessor from utils.py"""
    print("=" * 60)
    print("TESTING: PersonalDataSheetProcessor (Current System)")
    print("=" * 60)
    
    try:
        from utils import PersonalDataSheetProcessor
        processor = PersonalDataSheetProcessor()
        
        file_path = "Sample PDS Lenar.xlsx"
        if not os.path.exists(file_path):
            print(f"❌ Test file {file_path} not found")
            return None
            
        print(f"📄 Extracting from: {file_path}")
        extracted_data = processor.extract_pds_data(file_path)
        
        if extracted_data:
            print("✅ Extraction successful!")
            print(f"📊 Data sections found: {list(extracted_data.keys())}")
            
            # Show detailed breakdown
            for section, data in extracted_data.items():
                print(f"\n📋 {section.upper()}:")
                if isinstance(data, list):
                    print(f"   📦 Total entries: {len(data)}")
                    for i, entry in enumerate(data[:3], 1):  # Show first 3 entries
                        print(f"   [{i}] {entry}")
                        if i >= 3 and len(data) > 3:
                            print(f"   ... and {len(data) - 3} more entries")
                            break
                elif isinstance(data, dict):
                    print(f"   📦 Fields: {list(data.keys())}")
                    for key, value in list(data.items())[:5]:  # Show first 5 fields
                        print(f"   {key}: {value}")
                    if len(data) > 5:
                        print(f"   ... and {len(data) - 5} more fields")
                else:
                    print(f"   📦 Value: {data}")
                    
            return extracted_data
        else:
            print("❌ Extraction failed - no data returned")
            return None
            
    except Exception as e:
        print(f"❌ Error with PersonalDataSheetProcessor: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_pds_extractor():
    """Test the PDSExtractor from pds_extractor.py"""
    print("\n" + "=" * 60)
    print("TESTING: PDSExtractor (Dedicated PDS Processor)")
    print("=" * 60)
    
    try:
        from pds_extractor import PDSExtractor
        extractor = PDSExtractor()
        
        file_path = "Sample PDS Lenar.xlsx"
        if not os.path.exists(file_path):
            print(f"❌ Test file {file_path} not found")
            return None
            
        print(f"📄 Extracting from: {file_path}")
        extracted_data = extractor.extract_pds_data(file_path)
        
        if extracted_data:
            print("✅ Extraction successful!")
            print(f"📊 Data sections found: {list(extracted_data.keys())}")
            
            # Show detailed breakdown
            for section, data in extracted_data.items():
                print(f"\n📋 {section.upper()}:")
                if isinstance(data, list):
                    print(f"   📦 Total entries: {len(data)}")
                    for i, entry in enumerate(data[:3], 1):  # Show first 3 entries
                        if isinstance(entry, dict):
                            print(f"   [{i}] Fields: {list(entry.keys())}")
                            for key, value in list(entry.items())[:3]:
                                print(f"       {key}: {value}")
                        else:
                            print(f"   [{i}] {entry}")
                        if i >= 3 and len(data) > 3:
                            print(f"   ... and {len(data) - 3} more entries")
                            break
                elif isinstance(data, dict):
                    print(f"   📦 Fields: {list(data.keys())}")
                    for key, value in list(data.items())[:5]:  # Show first 5 fields
                        if isinstance(value, dict):
                            print(f"   {key}: {list(value.keys())}")
                        elif isinstance(value, list):
                            print(f"   {key}: [{len(value)} items]")
                        else:
                            print(f"   {key}: {value}")
                    if len(data) > 5:
                        print(f"   ... and {len(data) - 5} more fields")
                else:
                    print(f"   📦 Value: {data}")
            
            # Show errors and warnings
            if extractor.errors:
                print(f"\n⚠️ Errors ({len(extractor.errors)}):")
                for error in extractor.errors:
                    print(f"   - {error}")
                    
            if extractor.warnings:
                print(f"\n⚠️ Warnings ({len(extractor.warnings)}):")
                for warning in extractor.warnings:
                    print(f"   - {warning}")
                    
            return extracted_data
        else:
            print("❌ Extraction failed - no data returned")
            return None
            
    except Exception as e:
        print(f"❌ Error with PDSExtractor: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_improved_pds_extractor():
    """Test the ImprovedPDSExtractor from improved_pds_extractor.py"""
    print("\n" + "=" * 60)
    print("TESTING: ImprovedPDSExtractor (Advanced Multi-format)")
    print("=" * 60)
    
    try:
        from improved_pds_extractor import ImprovedPDSExtractor
        extractor = ImprovedPDSExtractor()
        
        file_path = "Sample PDS Lenar.xlsx"
        if not os.path.exists(file_path):
            print(f"❌ Test file {file_path} not found")
            return None
            
        print(f"📄 Extracting from: {file_path}")
        extracted_data = extractor.extract_pds_data(file_path)
        
        if extracted_data:
            print("✅ Extraction successful!")
            print(f"📊 Data sections found: {list(extracted_data.keys())}")
            
            # Show detailed breakdown
            for section, data in extracted_data.items():
                print(f"\n📋 {section.upper()}:")
                if isinstance(data, list):
                    print(f"   📦 Total entries: {len(data)}")
                    for i, entry in enumerate(data[:3], 1):  # Show first 3 entries
                        if isinstance(entry, dict):
                            print(f"   [{i}] Fields: {list(entry.keys())}")
                            for key, value in list(entry.items())[:3]:
                                print(f"       {key}: {value}")
                        else:
                            print(f"   [{i}] {entry}")
                        if i >= 3 and len(data) > 3:
                            print(f"   ... and {len(data) - 3} more entries")
                            break
                elif isinstance(data, dict):
                    print(f"   📦 Fields: {list(data.keys())}")
                    for key, value in list(data.items())[:5]:  # Show first 5 fields
                        if isinstance(value, dict):
                            print(f"   {key}: {list(value.keys())}")
                        elif isinstance(value, list):
                            print(f"   {key}: [{len(value)} items]")
                        else:
                            print(f"   {key}: {value}")
                    if len(data) > 5:
                        print(f"   ... and {len(data) - 5} more fields")
                else:
                    print(f"   📦 Value: {data}")
            
            # Show errors and warnings
            if extractor.errors:
                print(f"\n⚠️ Errors ({len(extractor.errors)}):")
                for error in extractor.errors:
                    print(f"   - {error}")
                    
            if extractor.warnings:
                print(f"\n⚠️ Warnings ({len(extractor.warnings)}):")
                for warning in extractor.warnings:
                    print(f"   - {warning}")
                    
            return extracted_data
        else:
            print("❌ Extraction failed - no data returned")
            return None
            
    except Exception as e:
        print(f"❌ Error with ImprovedPDSExtractor: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_personal_info(basic_data, pds_data, improved_data):
    """Compare personal information extraction across all extractors"""
    print("\n" + "=" * 80)
    print("🔍 DETAILED COMPARISON: PERSONAL INFORMATION")
    print("=" * 80)
    
    # Get personal info from each extractor
    basic_personal = {}
    pds_personal = {}
    improved_personal = {}
    
    # Extract personal info from each dataset
    if basic_data:
        # PersonalDataSheetProcessor doesn't have a dedicated personal_info section
        # It spreads this data across different sections
        print("📊 Basic Processor: No dedicated personal_info section")
        basic_personal = {"note": "Data spread across other sections"}
    
    if pds_data and 'personal_info' in pds_data:
        pds_personal = pds_data['personal_info']
        
    if improved_data and 'personal_info' in improved_data:
        improved_personal = improved_data['personal_info']
    
    # Compare key personal fields
    key_fields = [
        'surname', 'first_name', 'middle_name', 'name_extension',
        'date_of_birth', 'place_of_birth', 'sex', 'civil_status',
        'email', 'mobile_no', 'telephone_no', 'citizenship',
        'gsis_id', 'pagibig_id', 'philhealth_no', 'sss_no', 'tin_no'
    ]
    
    print(f"\n{'Field':<20} {'PDSExtractor':<25} {'ImprovedExtractor':<25}")
    print("-" * 70)
    
    for field in key_fields:
        pds_value = str(pds_personal.get(field, 'N/A'))[:20]
        improved_value = str(improved_personal.get(field, 'N/A'))[:20]
        print(f"{field:<20} {pds_value:<25} {improved_value:<25}")

def compare_education_data(basic_data, pds_data, improved_data):
    """Compare educational background extraction"""
    print("\n" + "=" * 80)
    print("🎓 DETAILED COMPARISON: EDUCATIONAL BACKGROUND")
    print("=" * 80)
    
    # Get education data from each extractor
    basic_education = basic_data.get('educational_background', []) if basic_data else []
    pds_education = pds_data.get('personal_info', {}).get('education', {}) if pds_data else {}
    improved_education = improved_data.get('educational_background', []) if improved_data else []
    
    print(f"📊 Basic Processor: {len(basic_education)} entries")
    for i, entry in enumerate(basic_education[:3], 1):
        print(f"   [{i}] {entry}")
    
    print(f"\n📊 PDS Extractor: {len(pds_education)} education levels")
    for level, info in list(pds_education.items())[:5]:
        print(f"   {level}: {info}")
    
    print(f"\n📊 Improved Extractor: {len(improved_education)} entries")
    for i, entry in enumerate(improved_education[:3], 1):
        if isinstance(entry, dict):
            level = entry.get('level', 'N/A')
            school = entry.get('school', 'N/A')
            degree = entry.get('degree_course', 'N/A')
            print(f"   [{i}] {level}: {school} - {degree}")
        else:
            print(f"   [{i}] {entry}")

def compare_work_experience(basic_data, pds_data, improved_data):
    """Compare work experience extraction"""
    print("\n" + "=" * 80)
    print("💼 DETAILED COMPARISON: WORK EXPERIENCE")
    print("=" * 80)
    
    # Get work experience from each extractor
    basic_work = basic_data.get('work_experience', []) if basic_data else []
    pds_work = pds_data.get('work_experience', []) if pds_data else []
    improved_work = improved_data.get('work_experience', []) if improved_data else []
    
    print(f"📊 Basic Processor: {len(basic_work)} entries")
    for i, entry in enumerate(basic_work[:3], 1):
        if isinstance(entry, dict):
            position = entry.get('position', 'N/A')
            company = entry.get('company', 'N/A')
            period = f"{entry.get('date_from', 'N/A')} to {entry.get('date_to', 'N/A')}"
            print(f"   [{i}] {position} at {company} ({period})")
        else:
            print(f"   [{i}] {entry}")
    
    print(f"\n📊 PDS Extractor: {len(pds_work)} entries")
    for i, entry in enumerate(pds_work[:3], 1):
        if isinstance(entry, dict):
            position = entry.get('position', 'N/A')
            company = entry.get('company', 'N/A')
            period = f"{entry.get('date_from', 'N/A')} to {entry.get('date_to', 'N/A')}"
            salary = entry.get('salary', 'N/A')
            print(f"   [{i}] {position} at {company} ({period}) - Salary: {salary}")
        else:
            print(f"   [{i}] {entry}")
    
    print(f"\n📊 Improved Extractor: {len(improved_work)} entries")
    for i, entry in enumerate(improved_work[:3], 1):
        if isinstance(entry, dict):
            position = entry.get('position', 'N/A')
            company = entry.get('company', 'N/A')
            period = f"{entry.get('period_from', 'N/A')} to {entry.get('period_to', 'N/A')}"
            salary = entry.get('monthly_salary', 'N/A')
            print(f"   [{i}] {position} at {company} ({period}) - Salary: {salary}")
        else:
            print(f"   [{i}] {entry}")

def compare_training_data(basic_data, pds_data, improved_data):
    """Compare training/learning development extraction"""
    print("\n" + "=" * 80)
    print("📚 DETAILED COMPARISON: TRAINING & DEVELOPMENT")
    print("=" * 80)
    
    # Get training data from each extractor
    basic_training = basic_data.get('learning_development', []) if basic_data else []
    pds_training = pds_data.get('training', []) if pds_data else []
    improved_training = improved_data.get('learning_development', []) if improved_data else []
    
    print(f"📊 Basic Processor: {len(basic_training)} entries")
    for i, entry in enumerate(basic_training[:3], 1):
        if isinstance(entry, dict):
            title = entry.get('title', 'N/A')
            hours = entry.get('hours', 'N/A')
            provider = entry.get('conductor', 'N/A')
            print(f"   [{i}] {title} ({hours} hours) - {provider}")
        else:
            print(f"   [{i}] {entry}")
    
    print(f"\n📊 PDS Extractor: {len(pds_training)} entries")
    for i, entry in enumerate(pds_training[:3], 1):
        if isinstance(entry, dict):
            title = entry.get('title', 'N/A')
            hours = entry.get('hours', 'N/A')
            provider = entry.get('conductor', 'N/A')
            training_type = entry.get('type', 'N/A')
            print(f"   [{i}] {title} ({hours} hours) - {provider} [Type: {training_type}]")
        else:
            print(f"   [{i}] {entry}")
    
    print(f"\n📊 Improved Extractor: {len(improved_training)} entries")
    for i, entry in enumerate(improved_training[:3], 1):
        if isinstance(entry, dict):
            title = entry.get('title', 'N/A')
            hours = entry.get('hours', 'N/A')
            provider = entry.get('conductor', 'N/A')
            training_type = entry.get('type', 'N/A')
            print(f"   [{i}] {title} ({hours} hours) - {provider} [Type: {training_type}]")
        else:
            print(f"   [{i}] {entry}")

def save_detailed_comparison(basic_data, pds_data, improved_data):
    """Save detailed comparison to JSON files"""
    print("\n" + "=" * 80)
    print("💾 SAVING DETAILED COMPARISON DATA")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save each extractor's results
    if basic_data:
        filename = f"basic_processor_results_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(basic_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ Basic Processor results saved to: {filename}")
    
    if pds_data:
        filename = f"pds_extractor_results_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(pds_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ PDS Extractor results saved to: {filename}")
    
    if improved_data:
        filename = f"improved_extractor_results_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(improved_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ Improved Extractor results saved to: {filename}")
    
    # Create comparison summary
    summary = {
        "comparison_timestamp": timestamp,
        "test_file": "Sample PDS Lenar.xlsx",
        "extractors_tested": [],
        "summary": {}
    }
    
    if basic_data:
        summary["extractors_tested"].append("PersonalDataSheetProcessor")
        summary["summary"]["basic_processor"] = {
            "sections": list(basic_data.keys()),
            "total_sections": len(basic_data),
            "educational_entries": len(basic_data.get('educational_background', [])),
            "work_entries": len(basic_data.get('work_experience', [])),
            "training_entries": len(basic_data.get('learning_development', []))
        }
    
    if pds_data:
        summary["extractors_tested"].append("PDSExtractor")
        summary["summary"]["pds_extractor"] = {
            "sections": list(pds_data.keys()),
            "total_sections": len(pds_data),
            "work_entries": len(pds_data.get('work_experience', [])),
            "training_entries": len(pds_data.get('training', [])),
            "eligibility_entries": len(pds_data.get('eligibility', [])),
            "voluntary_entries": len(pds_data.get('voluntary_work', []))
        }
    
    if improved_data:
        summary["extractors_tested"].append("ImprovedPDSExtractor")
        summary["summary"]["improved_extractor"] = {
            "sections": list(improved_data.keys()),
            "total_sections": len(improved_data),
            "educational_entries": len(improved_data.get('educational_background', [])),
            "work_entries": len(improved_data.get('work_experience', [])),
            "training_entries": len(improved_data.get('learning_development', [])),
            "voluntary_entries": len(improved_data.get('voluntary_work', []))
        }
    
    summary_filename = f"extractor_comparison_summary_{timestamp}.json"
    with open(summary_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ Comparison summary saved to: {summary_filename}")

def main():
    """Run comprehensive comparison of all PDS extractors"""
    print("🧪 PDS EXTRACTOR COMPARISON TEST")
    print("Testing all available extractors with Sample PDS Lenar.xlsx")
    print("=" * 80)
    
    # Test each extractor
    print("🚀 Running tests on all extractors...")
    
    basic_data = test_basic_pds_processor()
    pds_data = test_pds_extractor()
    improved_data = test_improved_pds_extractor()
    
    # Run detailed comparisons
    if any([basic_data, pds_data, improved_data]):
        print("\n" + "🔍 RUNNING DETAILED COMPARISONS...")
        
        compare_personal_info(basic_data, pds_data, improved_data)
        compare_education_data(basic_data, pds_data, improved_data)
        compare_work_experience(basic_data, pds_data, improved_data)
        compare_training_data(basic_data, pds_data, improved_data)
        
        # Save all results
        save_detailed_comparison(basic_data, pds_data, improved_data)
        
        # Final recommendation
        print("\n" + "=" * 80)
        print("🏆 FINAL COMPARISON SUMMARY")
        print("=" * 80)
        
        print("📊 EXTRACTION RESULTS:")
        extractors_working = []
        
        if basic_data:
            print(f"✅ PersonalDataSheetProcessor: {len(basic_data)} sections")
            extractors_working.append("Basic")
        else:
            print("❌ PersonalDataSheetProcessor: Failed")
            
        if pds_data:
            print(f"✅ PDSExtractor: {len(pds_data)} sections")
            extractors_working.append("PDS")
        else:
            print("❌ PDSExtractor: Failed")
            
        if improved_data:
            print(f"✅ ImprovedPDSExtractor: {len(improved_data)} sections")
            extractors_working.append("Improved")
        else:
            print("❌ ImprovedPDSExtractor: Failed")
        
        print(f"\n🎯 Working Extractors: {', '.join(extractors_working)}")
        
        if len(extractors_working) > 1:
            print(f"\n💡 RECOMMENDATION:")
            print(f"   Compare the detailed JSON output files to see which extractor")
            print(f"   provides the most complete and accurate data for your needs.")
            print(f"   Look at the structure, field names, and data completeness.")
        
    else:
        print("❌ All extractors failed! Check if Sample PDS Lenar.xlsx exists and is valid.")

if __name__ == "__main__":
    main()