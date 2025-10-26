#!/usr/bin/env python3
"""
Test if the LSPU assessment engine works with our candidate data
"""

import sys
import os
from utils import PersonalDataSheetProcessor
from assessment_engine import UniversityAssessmentEngine

def test_lspu_assessment():
    """Test if LSPU assessment works with our data"""
    print("🔍 Testing LSPU assessment engine...")
    
    test_file = "Sample PDS New.xlsx"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    # LSPU job like in the logs
    lspu_job = {
        'id': 2,
        'title': 'Instructor I',
        'source': 'LSPU',
        'position_type_id': 1  # Might be needed
    }
    
    try:
        print("\n1. Getting candidate data...")
        pds_processor = PersonalDataSheetProcessor()
        candidate_data = pds_processor.process_excel_pds_file(test_file, test_file, lspu_job)
        
        if not candidate_data or 'pds_data' not in candidate_data:
            print("❌ No valid candidate data")
            return False
        
        print("✅ Got candidate data with pds_data")
        pds_data = candidate_data['pds_data']
        
        print("\n2. Testing LSPU assessment engine...")
        try:
            from database import DatabaseManager
            db_manager = DatabaseManager()
            assessment_engine = UniversityAssessmentEngine(db_manager)
            print("✅ Assessment engine created with db_manager")
            
            print("\n3. Running LSPU assessment...")
            assessment_result = assessment_engine.assess_candidate_for_lspu_job(
                candidate_data=pds_data,
                lspu_job=lspu_job,
                position_type_id=lspu_job.get('position_type_id')
            )
            
            print("✅ LSPU assessment completed successfully")
            print(f"📊 Assessment result type: {type(assessment_result)}")
            
            if isinstance(assessment_result, dict):
                score = assessment_result.get('automated_score', 0)
                percentage_score = assessment_result.get('percentage_score', 0)
                print(f"📊 Automated score: {score}")
                print(f"📊 Percentage score: {percentage_score}")
                return True
            else:
                print("❌ Assessment result is not a dict")
                return False
                
        except Exception as assessment_error:
            print(f"❌ LSPU assessment failed: {assessment_error}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lspu_assessment()
    print(f"\n{'=' * 50}")
    if success:
        print("🎉 LSPU assessment test PASSED!")
    else:
        print("❌ LSPU assessment test FAILED!")
        print("🔧 This is likely where Flask processing fails")
    print(f"{'=' * 50}")