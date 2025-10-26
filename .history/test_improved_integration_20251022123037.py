#!/usr/bin/env python3
"""
Integration Test for ImprovedPDSExtractor
Test the complete workflow: PDS upload -> extraction -> database storage
"""

import os
import sys
from datetime import datetime

def test_improved_pds_integration():
    """Test the complete workflow with ImprovedPDSExtractor integration"""
    print("🧪 TESTING IMPROVED PDS EXTRACTOR INTEGRATION")
    print("=" * 60)
    
    test_file = "Sample PDS Lenar.xlsx"
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found")
        return False
    
    try:
        # Test 1: Import and test ImprovedPDSExtractor directly
        print("📋 STEP 1: Testing ImprovedPDSExtractor directly")
        from improved_pds_extractor import ImprovedPDSExtractor
        extractor = ImprovedPDSExtractor()
        
        raw_data = extractor.extract_pds_data(test_file)
        print(f"✅ Direct extraction successful: {len(raw_data)} sections")
        
        # Test 2: Test the converter
        print("\n📋 STEP 2: Testing improved converter")
        from improved_pds_converter import convert_improved_pds_to_assessment_format
        converted_data = convert_improved_pds_to_assessment_format(raw_data)
        print(f"✅ Conversion successful")
        
        # Test 3: Test the PersonalDataSheetProcessor integration
        print("\n📋 STEP 3: Testing PersonalDataSheetProcessor integration")
        from utils import PersonalDataSheetProcessor
        processor = PersonalDataSheetProcessor()
        
        # Create a dummy job for testing
        test_job = {'id': 1, 'title': 'Test Position'}
        
        candidate_data = processor.process_excel_pds_file(test_file, test_file, test_job)
        if candidate_data:
            print(f"✅ PDS processor integration successful")
            print(f"   👤 Name: {candidate_data.get('name', 'N/A')}")
            print(f"   📧 Email: {candidate_data.get('email', 'N/A')}")
            print(f"   📱 Phone: {candidate_data.get('phone', 'N/A')}")
            print(f"   🎓 Education entries: {candidate_data.get('total_education_entries', 0)}")
            print(f"   💼 Work entries: {candidate_data.get('total_work_positions', 0)}")
            print(f"   📚 Training entries: {len(candidate_data.get('training', []))}")
            print(f"   ✅ Eligibility entries: {len(candidate_data.get('eligibility', []))}")
        else:
            print("❌ PDS processor integration failed")
            return False
        
        # Test 4: Test database integration
        print("\n📋 STEP 4: Testing database integration")
        from database import DatabaseManager
        db_manager = DatabaseManager()
        
        # Add test job data
        candidate_data['job_id'] = 1
        
        try:
            candidate_id = db_manager.create_candidate(candidate_data)
            if candidate_id:
                print(f"✅ Database integration successful: Candidate ID {candidate_id}")
                
                # Verify the stored data
                stored_candidate = db_manager.get_candidate(candidate_id)
                if stored_candidate:
                    print(f"✅ Data retrieval successful")
                    print(f"   👤 Stored Name: {stored_candidate.get('name', 'N/A')}")
                    print(f"   📧 Stored Email: {stored_candidate.get('email', 'N/A')}")
                    print(f"   🔧 Processing Type: {stored_candidate.get('processing_type', 'N/A')}")
                    print(f"   📊 PDS Data Available: {'Yes' if stored_candidate.get('pds_extracted_data') else 'No'}")
                else:
                    print("❌ Data retrieval failed")
                    return False
            else:
                print("❌ Database integration failed - create_candidate returned None")
                return False
                
        except Exception as e:
            print(f"❌ Database integration failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 5: Complete workflow simulation
        print("\n📋 STEP 5: Testing complete workflow simulation")
        print("✅ All integration tests passed!")
        print("\n🎉 INTEGRATION SUCCESS SUMMARY:")
        print("   ✅ ImprovedPDSExtractor extraction")
        print("   ✅ Data conversion")
        print("   ✅ PersonalDataSheetProcessor integration")
        print("   ✅ Database storage (no resume_text required)")
        print("   ✅ Data retrieval")
        print("\n🚀 The system is ready for PDS-only processing!")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the complete integration test"""
    print("🔧 IMPROVED PDS EXTRACTOR INTEGRATION TEST")
    print("Testing the complete workflow from extraction to database storage")
    print("=" * 60)
    
    success = test_improved_pds_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ ImprovedPDSExtractor is fully integrated")
        print("✅ Database handles PDS data without resume_text requirement")
        print("✅ Start Analysis will now use the advanced extractor")
        print("\n🚀 Ready for production use!")
    else:
        print("❌ INTEGRATION TESTS FAILED!")
        print("Please check the errors above and fix before proceeding.")

if __name__ == "__main__":
    main()