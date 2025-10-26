#!/usr/bin/env python3
"""
Backend API functions for frontend integration
Reusable components from test_real_data_only.py
"""

import os
import sqlite3
from datetime import datetime
import pandas as pd

def process_candidate_assessment(pds_filepath, job_id):
    """
    Main function for frontend integration
    Process PDS file and return assessment results
    
    Args:
        pds_filepath: Path to uploaded PDS file
        job_id: ID of the job posting for assessment
        
    Returns:
        dict: Assessment results with scores and recommendations
    """
    try:
        # Step 1: Get job posting
        job_data = get_lspu_job_by_id(job_id)
        if not job_data:
            return {'error': f'Job ID {job_id} not found'}
        
        # Step 2: Extract PDS data
        pds_data = extract_pds_from_file(pds_filepath)
        if not pds_data:
            return {'error': 'Failed to extract PDS data'}
        
        # Step 3: Run assessment
        assessment_result = run_assessment_engine(pds_data, job_data)
        if not assessment_result:
            return {'error': 'Assessment failed'}
        
        # Step 4: Format response for frontend
        return format_assessment_response(assessment_result, pds_data, job_data)
        
    except Exception as e:
        return {'error': f'Processing failed: {str(e)}'}

def get_lspu_job_by_id(job_id):
    """Get LSPU job posting by ID"""
    try:
        conn = sqlite3.connect('resume_screening.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT jp.id, jp.job_reference_number, jp.position_title, 
                   jp.education_requirements, jp.experience_requirements,
                   jp.training_requirements, jp.eligibility_requirements,
                   jp.special_requirements, jp.salary_grade, jp.status,
                   cl.campus_name, pt.name as position_type_name
            FROM lspu_job_postings jp
            LEFT JOIN campus_locations cl ON jp.campus_id = cl.id  
            LEFT JOIN position_types pt ON jp.position_type_id = pt.id
            WHERE jp.id = ? AND jp.status = 'published'
        """, (job_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'job_reference_number': row[1],
                'position_title': row[2],
                'education_requirements': row[3],
                'experience_requirements': row[4],
                'training_requirements': row[5],
                'eligibility_requirements': row[6],
                'special_requirements': row[7],
                'salary_grade': row[8],
                'status': row[9],
                'campus_name': row[10],
                'position_type_name': row[11]
            }
        return None
        
    except Exception as e:
        print(f"Error getting job: {e}")
        return None

def extract_pds_from_file(filepath):
    """Extract PDS data from uploaded file"""
    try:
        from utils import PersonalDataSheetProcessor
        
        pds_processor = PersonalDataSheetProcessor()
        extracted_data = pds_processor.extract_pds_data(filepath)
        
        # Convert to assessment format
        from test_real_data_only import convert_pds_to_assessment_format
        converted_data = convert_pds_to_assessment_format(extracted_data, filepath)
        
        return converted_data
        
    except Exception as e:
        print(f"Error extracting PDS: {e}")
        return None

def run_assessment_engine(pds_data, job_data):
    """Run the assessment engine"""
    try:
        from database import DatabaseManager
        from assessment_engine import UniversityAssessmentEngine
        
        db_manager = DatabaseManager()
        assessment_engine = UniversityAssessmentEngine(db_manager)
        
        assessment_result = assessment_engine.assess_candidate_for_lspu_job(
            candidate_data=pds_data,
            lspu_job=job_data,
            position_type_id=job_data.get('position_type_id', 1)
        )
        
        return assessment_result
        
    except Exception as e:
        print(f"Error in assessment: {e}")
        return None

def format_assessment_response(assessment_result, pds_data, job_data):
    """Format assessment results for frontend consumption"""
    try:
        response = {
            'success': True,
            'candidate': {
                'name': pds_data['basic_info'].get('name', 'Unknown'),
                'email': pds_data['basic_info'].get('email', 'N/A'),
                'phone': pds_data['basic_info'].get('phone', 'N/A')
            },
            'job': {
                'title': job_data.get('position_title', 'Unknown'),
                'reference': job_data.get('job_reference_number', 'N/A'),
                'salary_grade': job_data.get('salary_grade', 'N/A')
            },
            'assessment': {
                'total_score': assessment_result.get('automated_score', 0),
                'percentage': assessment_result.get('percentage_score', 0),
                'max_score': 85,
                'recommendation': assessment_result.get('recommendation', 'unknown'),
                'assessment_date': datetime.now().isoformat()
            },
            'category_scores': {},
            'data_summary': {
                'education_entries': len(pds_data.get('education', [])),
                'work_experience': len(pds_data.get('experience', [])),
                'training_programs': len(pds_data.get('training', [])),
                'eligibilities': len(pds_data.get('eligibility', [])),
                'volunteer_work': len(pds_data.get('volunteer_work', []))
            }
        }
        
        # Add detailed category breakdown
        assessment_results = assessment_result.get('assessment_results', {})
        for category, details in assessment_results.items():
            response['category_scores'][category] = {
                'score': details.get('score', 0),
                'max_possible': details.get('max_possible', 0),
                'weight_percentage': details.get('category_weight', 0),
                'details': details.get('details', {})
            }
        
        return response
        
    except Exception as e:
        return {'error': f'Failed to format response: {str(e)}'}

def get_available_jobs():
    """Get list of available job postings for frontend dropdown"""
    try:
        conn = sqlite3.connect('resume_screening.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, job_reference_number, position_title, salary_grade
            FROM lspu_job_postings 
            WHERE status = 'published'
            ORDER BY position_title
        """)
        
        jobs = []
        for row in cursor.fetchall():
            jobs.append({
                'id': row[0],
                'reference': row[1],
                'title': row[2],
                'salary_grade': row[3]
            })
        
        conn.close()
        return jobs
        
    except Exception as e:
        print(f"Error getting jobs: {e}")
        return []

if __name__ == "__main__":
    # Test the API functions
    print("🧪 Testing Backend API Functions")
    
    # Test job listing
    jobs = get_available_jobs()
    print(f"✅ Found {len(jobs)} available jobs")
    
    # Test with sample file if exists
    sample_file = "Sample PDS New.xlsx"
    if os.path.exists(sample_file) and jobs:
        print(f"\n🔍 Testing assessment with {sample_file}")
        result = process_candidate_assessment(sample_file, jobs[1]['id'])
        
        if result.get('success'):
            print("✅ Assessment API working!")
            print(f"   Score: {result['assessment']['total_score']:.2f}")
            print(f"   Recommendation: {result['assessment']['recommendation']}")
        else:
            print(f"❌ Assessment failed: {result.get('error')}")