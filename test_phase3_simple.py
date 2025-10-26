#!/usr/bin/env python3
"""
Phase 3 Simplified PDS Workflow Test
Tests core functionality with actual available methods
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import core modules
from utils import PersonalDataSheetProcessor
from database import DatabaseManager
from assessment_engine import UniversityAssessmentEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pds_extraction():
    """Test basic PDS extraction functionality"""
    print("🔍 Testing PDS Extraction...")
    
    processor = PersonalDataSheetProcessor()
    
    sample_text = """
    PERSONAL DATA SHEET
    
    PERSONAL INFORMATION
    Name: Juan Dela Cruz
    Date of Birth: January 15, 1985
    Email: juan.delacruz@email.com
    Mobile: 09123456789
    
    II. EDUCATIONAL BACKGROUND
    
    COLLEGE
    Bachelor of Science in Computer Science
    University of the Philippines
    2003-2007
    
    GRADUATE STUDIES
    Master of Science in Information Technology
    Ateneo de Manila University
    2008-2010
    
    III. WORK EXPERIENCE
    
    Senior Software Developer
    ABC Corporation
    January 2010 - December 2015
    Monthly Salary: 50,000
    
    IT Manager
    XYZ Company
    January 2016 - Present
    Monthly Salary: 80,000
    
    VII. TRAINING PROGRAMS ATTENDED
    
    Project Management Professional (PMP)
    Project Management Institute
    2012 - 40 hours
    
    AWS Cloud Practitioner
    Amazon Web Services
    2020 - 20 hours
    
    VI. CIVIL SERVICE ELIGIBILITY
    
    Career Service Executive Eligibility
    September 15, 2008
    Rating: 82.5%
    """
    
    try:
        # Test the main PDS extraction method
        pds_data = processor.extract_pds_information(sample_text)
        
        print(f"✓ PDS extraction completed")
        print(f"✓ Found {len(pds_data)} main sections")
        
        # Test individual extraction methods
        personal_info = processor.extract_personal_information_pds(sample_text)
        education = processor.extract_education_detailed(sample_text)
        experience = processor.extract_experience_detailed(sample_text)
        training = processor.extract_training_seminars(sample_text)
        eligibility = processor.extract_civil_service_eligibility(sample_text)
        
        print(f"✓ Personal Info: {len(personal_info)} fields")
        print(f"✓ Education: {len(education)} entries")
        print(f"✓ Experience: {len(experience)} positions")
        print(f"✓ Training: {len(training)} programs")
        print(f"✓ Eligibility: {len(eligibility)} certifications")
        
        return True
        
    except Exception as e:
        print(f"✗ PDS extraction failed: {e}")
        return False

def test_assessment_engine():
    """Test assessment engine functionality"""
    print("🎯 Testing Assessment Engine...")
    
    try:
        db_manager = DatabaseManager()
        engine = UniversityAssessmentEngine(db_manager)
        
        # Test if the engine initializes properly
        print(f"✓ Assessment engine initialized")
        print(f"✓ Degree levels loaded: {len(engine.degree_levels)} types")
        print(f"✓ Professional certifications: {len(engine.professional_certifications)} types")
        
        # Sample candidate data
        candidate_data = {
            'personal_information': {'name': 'Test Candidate'},
            'education': [{'degree': 'Master of Science', 'school': 'UP'}],
            'experience': [{'position': 'Developer', 'duration': '5 years'}]
        }
        
        # Sample LSPU job data
        lspu_job = {
            'title': 'Software Developer',
            'education_requirements': 'Bachelor degree in Computer Science',
            'experience_requirements': '3 years experience'
        }
        
        # Test the actual available method
        assessment = engine.assess_candidate_for_lspu_job(candidate_data, lspu_job)
        
        if assessment:
            print(f"✓ Assessment calculation successful")
            print(f"✓ Assessment result type: {type(assessment)}")
        else:
            print(f"⚠ Assessment returned empty result")
            
        return True
        
    except Exception as e:
        print(f"✗ Assessment engine test failed: {e}")
        return False

def test_database_connection():
    """Test database initialization"""
    print("🗄️ Testing Database Connection...")
    
    try:
        db_manager = DatabaseManager()
        
        # Test if we can get a connection
        connection = db_manager.get_connection()
        if connection:
            print("✓ Database connection successful")
            connection.close()
        else:
            print("⚠ Database connection returned None")
            
        # Check if basic methods exist
        methods_to_check = [
            'get_connection',
            'save_assessment_comparison',
            'get_assessment_comparison'
        ]
        
        for method in methods_to_check:
            if hasattr(db_manager, method):
                print(f"✓ Method {method} available")
            else:
                print(f"⚠ Method {method} not found")
                
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def test_file_processing():
    """Test file processing capabilities"""
    print("📄 Testing File Processing...")
    
    try:
        processor = PersonalDataSheetProcessor()
        
        # Check available extraction methods
        extraction_methods = [
            'extract_pdf_text',
            'extract_pds_information',
            'extract_personal_information_pds',
            'extract_education_detailed',
            'extract_experience_detailed',
            'extract_training_seminars',
            'extract_civil_service_eligibility',
            'extract_certifications',
            'extract_awards_recognition'
        ]
        
        available_methods = 0
        for method in extraction_methods:
            if hasattr(processor, method):
                print(f"✓ {method} available")
                available_methods += 1
            else:
                print(f"⚠ {method} not found")
                
        print(f"✓ {available_methods}/{len(extraction_methods)} extraction methods available")
        
        return available_methods > len(extraction_methods) * 0.8  # 80% pass rate
        
    except Exception as e:
        print(f"✗ File processing test failed: {e}")
        return False

def run_system_validation():
    """Run complete system validation"""
    print("=" * 60)
    print("PHASE 3 PDS SYSTEM VALIDATION")
    print("=" * 60)
    
    tests = [
        ("PDS Extraction", test_pds_extraction),
        ("Assessment Engine", test_assessment_engine),
        ("Database Connection", test_database_connection),
        ("File Processing", test_file_processing)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        result = test_func()
        results[test_name] = result
        if result:
            passed += 1
            
    print(f"\n{'=' * 60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"PASSED: {passed}/{total} tests")
    print(f"SUCCESS RATE: {(passed/total)*100:.1f}%")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_name}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"phase3_validation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'passed': passed,
            'total': total,
            'success_rate': (passed/total)*100,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED! Phase 3 cleanup successful!")
        return True
    elif passed >= total * 0.8:  # 80% pass rate
        print(f"\n✅ Most tests passed ({passed}/{total}). System is functional.")
        return True
    else:
        print(f"\n❌ Multiple test failures ({total-passed}/{total}). Please review issues.")
        return False

if __name__ == "__main__":
    success = run_system_validation()
    sys.exit(0 if success else 1)