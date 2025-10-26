#!/usr/bin/env python3
"""
Phase 2 Validation Test - Verify PDS system works after ResumeProcessor removal
Tests the core PDS functionality and ensures no broken dependencies.
"""

import sys
import os
import traceback
from pathlib import Path

def test_imports():
    """Test that all required imports work"""
    try:
        from utils import PersonalDataSheetProcessor
        print("✅ PersonalDataSheetProcessor import successful")
        
        from app import PDSAssessmentApp  
        print("✅ PDSAssessmentApp import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_pds_processor_initialization():
    """Test that PDS processor can be initialized"""
    try:
        from utils import PersonalDataSheetProcessor
        processor = PersonalDataSheetProcessor()
        
        # Check that essential attributes exist
        assert hasattr(processor, 'logger'), "Missing logger attribute"
        assert hasattr(processor, 'semantic_analyzer'), "Missing semantic_analyzer attribute"
        assert hasattr(processor, 'bias_detector'), "Missing bias_detector attribute"
        assert hasattr(processor, 'skills_dict'), "Missing skills_dict attribute"
        
        print("✅ PDS processor initialized successfully with all required attributes")
        return True
    except Exception as e:
        print(f"❌ PDS processor initialization failed: {e}")
        traceback.print_exc()
        return False

def test_app_initialization():
    """Test that the app can be initialized"""
    try:
        from app import PDSAssessmentApp
        app_instance = PDSAssessmentApp()
        
        # Check essential attributes
        assert hasattr(app_instance, 'pds_processor'), "Missing pds_processor attribute"
        assert hasattr(app_instance, 'processor'), "Missing processor attribute"
        assert app_instance.processor is app_instance.pds_processor, "Processor alias not set correctly"
        
        print("✅ PDSAssessmentApp initialized successfully")
        return True
    except Exception as e:
        print(f"❌ App initialization failed: {e}")
        traceback.print_exc()
        return False

def test_pds_functionality():
    """Test core PDS functionality"""
    try:
        from utils import PersonalDataSheetProcessor
        processor = PersonalDataSheetProcessor()
        
        # Test sample PDS content
        sample_pds_text = """
        PERSONAL DATA SHEET
        Name: Juan dela Cruz
        Email: juan.delacruz@email.com
        Phone: +63 912 345 6789
        
        EDUCATIONAL BACKGROUND:
        Bachelor of Science in Computer Science
        University of the Philippines
        2015-2019
        
        WORK EXPERIENCE:
        Software Developer
        ABC Company
        2019-2023
        
        SPECIAL SKILLS:
        Python, JavaScript, Database Management
        """
        
        # Test basic info extraction
        basic_info = processor.extract_basic_info(sample_pds_text)
        assert basic_info.get('name'), "Failed to extract name"
        assert basic_info.get('email'), "Failed to extract email" 
        assert basic_info.get('phone'), "Failed to extract phone"
        
        # Test education extraction
        education = processor.extract_education_pds(sample_pds_text)
        assert education, "Failed to extract education"
        
        # Test skills extraction
        skills = processor.extract_skills_pds(sample_pds_text)
        assert skills, "Failed to extract skills"
        
        print("✅ Core PDS functionality working correctly")
        return True
    except Exception as e:
        print(f"❌ PDS functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_no_resume_processor_references():
    """Test that ResumeProcessor is completely removed"""
    try:
        # Try to import ResumeProcessor - this should fail
        try:
            from utils import ResumeProcessor
            print("❌ ResumeProcessor still exists in utils!")
            return False
        except ImportError:
            print("✅ ResumeProcessor successfully removed from utils")
        
        # Check app.py doesn't reference ResumeProcessor
        with open('app.py', 'r', encoding='utf-8') as f:
            app_content = f.read()
            
        if 'ResumeProcessor' in app_content:
            print("❌ ResumeProcessor references still exist in app.py")
            return False
        else:
            print("✅ No ResumeProcessor references found in app.py")
        
        return True
    except Exception as e:
        print(f"❌ ResumeProcessor reference test failed: {e}")
        traceback.print_exc()
        return False

def run_validation_tests():
    """Run all validation tests"""
    print("🧪 Running Phase 2 Validation Tests")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("PDS Processor Initialization", test_pds_processor_initialization), 
        ("App Initialization", test_app_initialization),
        ("PDS Functionality", test_pds_functionality),
        ("No Resume References", test_no_resume_processor_references)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️ {test_name} failed!")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 2 cleanup successful!")
        print("✨ PDS system is fully functional without ResumeProcessor dependencies")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1)