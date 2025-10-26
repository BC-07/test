#!/usr/bin/env python3
"""
Phase 3 Complete PDS Workflow Test
Tests the full PDS assessment system end-to-end after legacy cleanup
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDSWorkflowTester:
    """Complete PDS workflow testing class"""
    
    def __init__(self):
        """Initialize the tester with core components"""
        self.pds_processor = PersonalDataSheetProcessor()
        self.db_manager = DatabaseManager()
        self.assessment_engine = UniversityAssessmentEngine(self.db_manager)
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': []
        }
        
    def test_pds_extraction(self, test_file_path: str = None) -> dict:
        """Test PDS extraction functionality"""
        test_name = "PDS Extraction Test"
        logger.info(f"Starting {test_name}")
        
        result = {
            'test_name': test_name,
            'status': 'PASS',
            'details': {},
            'errors': []
        }
        
        try:
            # Test basic PDS information extraction
            sample_pds_text = """
            PERSONAL DATA SHEET
            
            Name: Juan dela Cruz
            Date of Birth: January 15, 1985
            Email: juan.delacruz@email.com
            Phone: 09123456789
            
            EDUCATIONAL BACKGROUND:
            Bachelor of Science in Computer Science
            University of the Philippines - 2007
            
            Master of Science in Information Technology  
            Ateneo de Manila University - 2010
            
            WORK EXPERIENCE:
            Senior Software Developer
            ABC Corporation (2010-2015)
            - Led development of enterprise applications
            
            IT Manager
            XYZ Company (2015-Present)
            - Managed IT infrastructure and team
            
            TRAINING AND DEVELOPMENT:
            Project Management Certification - 2012
            AWS Cloud Practitioner - 2020
            
            CIVIL SERVICE ELIGIBILITY:
            Career Service Executive Eligibility - 2008
            """
            
            # Extract PDS information
            pds_data = self.pds_processor.extract_pds_information(sample_pds_text)
            
            # Validate extraction results
            required_sections = [
                'personal_information',
                'educational_background', 
                'work_experience',
                'training_and_development',
                'civil_service_eligibility'
            ]
            
            for section in required_sections:
                if section not in pds_data:
                    result['status'] = 'FAIL'
                    result['errors'].append(f"Missing section: {section}")
                else:
                    result['details'][section] = f"✓ Found {len(pds_data.get(section, []))} items"
            
            # Check personal information extraction
            personal_info = pds_data.get('personal_information', {})
            if personal_info.get('name'):
                result['details']['name_extraction'] = f"✓ Name: {personal_info['name']}"
            else:
                result['errors'].append("Failed to extract name")
                
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Exception: {str(e)}")
            logger.error(f"PDS extraction test failed: {e}")
            
        return result
    
    def test_assessment_engine(self) -> dict:
        """Test assessment engine functionality"""
        test_name = "Assessment Engine Test"
        logger.info(f"Starting {test_name}")
        
        result = {
            'test_name': test_name,
            'status': 'PASS',
            'details': {},
            'errors': []
        }
        
        try:
            # Sample PDS data for assessment
            sample_pds = {
                'personal_information': {
                    'name': 'Test Candidate',
                    'email': 'test@email.com'
                },
                'educational_background': [
                    {
                        'degree': 'Master of Science in Computer Science',
                        'school': 'University of the Philippines',
                        'year_graduated': '2015'
                    }
                ],
                'work_experience': [
                    {
                        'position': 'Senior Software Developer',
                        'company': 'Tech Corp',
                        'duration': '5 years'
                    }
                ],
                'civil_service_eligibility': [
                    {
                        'eligibility': 'Career Service Executive',
                        'date_taken': '2010'
                    }
                ]
            }
            
            # Sample job requirements
            job_requirements = {
                'education_requirements': {
                    'minimum_degree': 'Bachelor',
                    'preferred_degree': 'Master',
                    'field_of_study': ['Computer Science', 'Information Technology']
                },
                'experience_requirements': {
                    'minimum_years': 3,
                    'relevant_fields': ['Software Development', 'IT Management']
                }
            }
            
            # Test assessment calculation
            assessment_score = self.assessment_engine.calculate_candidate_score(
                sample_pds, job_requirements
            )
            
            if assessment_score and 'total_score' in assessment_score:
                result['details']['assessment_calculation'] = f"✓ Score calculated: {assessment_score['total_score']}"
                result['details']['score_breakdown'] = assessment_score.get('breakdown', {})
            else:
                result['status'] = 'FAIL'
                result['errors'].append("Failed to calculate assessment score")
                
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Exception: {str(e)}")
            logger.error(f"Assessment engine test failed: {e}")
            
        return result
    
    def test_database_operations(self) -> dict:
        """Test database operations"""
        test_name = "Database Operations Test"
        logger.info(f"Starting {test_name}")
        
        result = {
            'test_name': test_name,
            'status': 'PASS',
            'details': {},
            'errors': []
        }
        
        try:
            # Test database connection
            if self.db_manager.test_connection():
                result['details']['connection'] = "✓ Database connection successful"
            else:
                result['status'] = 'FAIL'
                result['errors'].append("Database connection failed")
                return result
            
            # Test candidate insertion (if method exists)
            if hasattr(self.db_manager, 'save_candidate_pds'):
                test_candidate = {
                    'name': 'Test User',
                    'email': 'test@example.com',
                    'pds_data': {'test': 'data'}
                }
                
                candidate_id = self.db_manager.save_candidate_pds(test_candidate)
                if candidate_id:
                    result['details']['data_insertion'] = f"✓ Test candidate saved with ID: {candidate_id}"
                    
                    # Clean up test data
                    if hasattr(self.db_manager, 'delete_candidate'):
                        self.db_manager.delete_candidate(candidate_id)
                        result['details']['cleanup'] = "✓ Test data cleaned up"
                else:
                    result['errors'].append("Failed to save test candidate")
            else:
                result['details']['note'] = "save_candidate_pds method not available for testing"
                
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Exception: {str(e)}")
            logger.error(f"Database operations test failed: {e}")
            
        return result
    
    def test_file_processing(self) -> dict:
        """Test file processing capabilities"""
        test_name = "File Processing Test"
        logger.info(f"Starting {test_name}")
        
        result = {
            'test_name': test_name,
            'status': 'PASS',
            'details': {},
            'errors': []
        }
        
        try:
            # Test PDF text extraction
            if hasattr(self.pds_processor, 'extract_pdf_text'):
                result['details']['pdf_extraction'] = "✓ PDF extraction method available"
            else:
                result['errors'].append("PDF extraction method not found")
            
            # Test DOCX text extraction
            if hasattr(self.pds_processor, 'extract_docx_text'):
                result['details']['docx_extraction'] = "✓ DOCX extraction method available"
            else:
                result['errors'].append("DOCX extraction method not found")
            
            # Test text processing methods
            processing_methods = [
                'extract_basic_info',
                'extract_education_pds',
                'extract_experience_pds',
                'extract_skills_pds'
            ]
            
            for method in processing_methods:
                if hasattr(self.pds_processor, method):
                    result['details'][f'{method}_available'] = f"✓ {method} method found"
                else:
                    result['errors'].append(f"{method} method not found")
                    
            if result['errors']:
                result['status'] = 'FAIL'
                
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Exception: {str(e)}")
            logger.error(f"File processing test failed: {e}")
            
        return result
    
    def run_complete_workflow_test(self) -> dict:
        """Run complete PDS workflow test"""
        logger.info("Starting Complete PDS Workflow Test")
        
        workflow_result = {
            'test_name': 'Complete PDS Workflow',
            'status': 'PASS',
            'details': {},
            'errors': []
        }
        
        try:
            # Step 1: PDS Extraction
            extraction_test = self.test_pds_extraction()
            workflow_result['details']['extraction'] = extraction_test
            
            # Step 2: Assessment Engine
            assessment_test = self.test_assessment_engine()
            workflow_result['details']['assessment'] = assessment_test
            
            # Step 3: Database Operations
            database_test = self.test_database_operations()
            workflow_result['details']['database'] = database_test
            
            # Step 4: File Processing
            file_test = self.test_file_processing()
            workflow_result['details']['file_processing'] = file_test
            
            # Determine overall status
            all_tests = [extraction_test, assessment_test, database_test, file_test]
            failed_tests = [test for test in all_tests if test['status'] == 'FAIL']
            
            if failed_tests:
                workflow_result['status'] = 'FAIL'
                workflow_result['errors'] = [f"Failed tests: {len(failed_tests)}/{len(all_tests)}"]
            else:
                workflow_result['details']['summary'] = f"✓ All {len(all_tests)} component tests passed"
                
        except Exception as e:
            workflow_result['status'] = 'FAIL'
            workflow_result['errors'].append(f"Workflow exception: {str(e)}")
            logger.error(f"Complete workflow test failed: {e}")
            
        return workflow_result
    
    def run_all_tests(self):
        """Run all PDS workflow tests"""
        logger.info("="*60)
        logger.info("PHASE 3 PDS WORKFLOW VALIDATION")
        logger.info("="*60)
        
        # Individual component tests
        tests = [
            self.test_pds_extraction(),
            self.test_assessment_engine(),
            self.test_database_operations(),
            self.test_file_processing()
        ]
        
        # Complete workflow test
        workflow_test = self.run_complete_workflow_test()
        tests.append(workflow_test)
        
        # Compile results
        self.test_results['tests'] = tests
        
        # Summary
        passed_tests = [test for test in tests if test['status'] == 'PASS']
        failed_tests = [test for test in tests if test['status'] == 'FAIL']
        
        logger.info(f"Test Results Summary:")
        logger.info(f"PASSED: {len(passed_tests)}")
        logger.info(f"FAILED: {len(failed_tests)}")
        
        for test in tests:
            status_icon = "✓" if test['status'] == 'PASS' else "✗"
            logger.info(f"{status_icon} {test['test_name']}: {test['status']}")
            
            if test['errors']:
                for error in test['errors']:
                    logger.error(f"  - {error}")
        
        # Save results
        results_file = f"phase3_pds_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"Detailed results saved to: {results_file}")
        
        return len(failed_tests) == 0

if __name__ == "__main__":
    tester = PDSWorkflowTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✓ ALL PDS WORKFLOW TESTS PASSED")
        print("Phase 3 backend cleanup completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Please review the errors and fix issues before proceeding.")
        sys.exit(1)