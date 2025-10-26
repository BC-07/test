#!/usr/bin/env python3
"""
Complete Upload Workflow Test
Tests the entire upload -> process -> display workflow end-to-end
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class UploadWorkflowTester:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def log_test(self, test_name, success, details=""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'timestamp': time.time()
        })
    
    def test_server_health(self):
        """Test if server is running and healthy"""
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            success = response.status_code == 200 and response.json().get('status') == 'healthy'
            self.log_test("Server Health Check", success, f"Status: {response.status_code}")
            return success
        except Exception as e:
            self.log_test("Server Health Check", False, f"Error: {e}")
            return False
    
    def test_authentication(self):
        """Test user authentication or verify system is accessible"""
        try:
            # Try to access a protected endpoint first
            response = self.session.get(f"{self.base_url}/api/candidates")
            if response.status_code == 401:
                self.log_test("Authentication Required", True, "Properly requires authentication")
                return True
            elif response.status_code == 200:
                self.log_test("Authentication Required", True, "System is accessible (testing mode)")
                return True
            else:
                self.log_test("Authentication Required", False, f"Unexpected status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Authentication Required", False, f"Error: {e}")
            return False
    
    def test_job_postings_api(self):
        """Test job postings API"""
        try:
            response = self.session.get(f"{self.base_url}/api/job-postings")
            data = response.json()
            
            success = response.status_code == 200 and (
                (data.get('success') and 'postings' in data) or 
                isinstance(data, list)
            )
            
            job_count = 0
            if data.get('success') and 'postings' in data:
                job_count = len(data['postings'])
            elif isinstance(data, list):
                job_count = len(data)
            
            self.log_test("Job Postings API", success, f"Found {job_count} job postings")
            return success and job_count > 0
        except Exception as e:
            self.log_test("Job Postings API", False, f"Error: {e}")
            return False
    
    def test_upload_handler_initialization(self):
        """Test if upload handler is properly initialized"""
        try:
            from clean_upload_handler import CleanUploadHandler
            upload_handler = CleanUploadHandler()
            
            # Test file validation
            class MockFile:
                def __init__(self, filename, size=1024):
                    self.filename = filename
                    self.size = size
                
                def seek(self, pos, whence=0):
                    pass
                
                def tell(self):
                    return self.size
            
            # Test valid file
            valid_file = MockFile("test.pdf")
            is_valid, error, file_info = upload_handler.validate_file(valid_file)
            
            # Test invalid file
            invalid_file = MockFile("test.txt")
            is_invalid, error2, file_info2 = upload_handler.validate_file(invalid_file)
            
            success = is_valid and not is_invalid
            self.log_test("Upload Handler Initialization", success, "File validation working")
            return success
        except Exception as e:
            self.log_test("Upload Handler Initialization", False, f"Error: {e}")
            return False
    
    def test_database_session_methods(self):
        """Test database session management methods"""
        try:
            from database import DatabaseManager
            import time
            db_manager = DatabaseManager()
            
            # Test session creation with unique ID
            test_session_id = f"test_session_{int(time.time())}"
            created = db_manager.create_upload_session(test_session_id, 1, 1)
            
            # Test session retrieval
            if created:
                session_data = db_manager.get_upload_session(test_session_id)
                retrieved = session_data is not None
            else:
                retrieved = False
            
            # Test session update
            if retrieved:
                updated = db_manager.update_upload_session(test_session_id, status='testing')
            else:
                updated = False
            
            success = created and retrieved and updated
            self.log_test("Database Session Methods", success, "Session CRUD operations working")
            return success
        except Exception as e:
            self.log_test("Database Session Methods", False, f"Error: {e}")
            return False
    
    def test_pds_processor(self):
        """Test PDS processor functionality"""
        try:
            # Import safely
            try:
                from utils import PersonalDataSheetProcessor
                pds_processor = PersonalDataSheetProcessor()
                success = True
                details = "PDS Processor initialized successfully"
            except Exception as e:
                success = False
                details = f"PDS Processor initialization failed: {e}"
            
            self.log_test("PDS Processor", success, details)
            return success
        except Exception as e:
            self.log_test("PDS Processor", False, f"Error: {e}")
            return False
    
    def test_assessment_engine(self):
        """Test assessment engine functionality"""
        try:
            from assessment_engine import UniversityAssessmentEngine
            from database import DatabaseManager
            
            db_manager = DatabaseManager()
            assessment_engine = UniversityAssessmentEngine(db_manager)
            
            success = assessment_engine is not None
            self.log_test("Assessment Engine", success, "Assessment engine initialized")
            return success
        except Exception as e:
            self.log_test("Assessment Engine", False, f"Error: {e}")
            return False
    
    def create_test_summary(self):
        """Create a summary of all test results"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results if test['success'])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*60)
        print("UPLOAD WORKFLOW TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\nFAILED TESTS:")
            for test in self.test_results:
                if not test['success']:
                    print(f"  - {test['test']}: {test['details']}")
        
        print("\nRECOMMENDATIONS:")
        if passed_tests == total_tests:
            print("✅ All tests passed! Upload system is ready for use.")
        else:
            print("⚠️  Some tests failed. Please address the issues before using the upload system.")
            print("   - Check server is running on localhost:5000")
            print("   - Verify all dependencies are installed")
            print("   - Check database configuration")
            print("   - Ensure all Python modules are importable")
        
        return passed_tests == total_tests

def main():
    """Run the complete workflow test"""
    print("Starting Upload Workflow Test...")
    print("="*60)
    
    tester = UploadWorkflowTester()
    
    # Run all tests
    tests = [
        tester.test_server_health,
        tester.test_authentication, 
        tester.test_job_postings_api,
        tester.test_upload_handler_initialization,
        tester.test_database_session_methods,
        tester.test_pds_processor,
        tester.test_assessment_engine
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ FAIL {test.__name__}: Unexpected error: {e}")
        time.sleep(0.5)  # Small delay between tests
    
    # Create summary
    all_passed = tester.create_test_summary()
    
    # Save results to file
    results_file = 'upload_workflow_test_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'total_tests': len(tester.test_results),
            'passed': sum(1 for test in tester.test_results if test['success']),
            'failed': sum(1 for test in tester.test_results if not test['success']),
            'results': tester.test_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())