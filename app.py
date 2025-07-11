from flask import Flask, request, render_template, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
from typing import Dict, List, Any, Optional
import pickle
from utils import ResumeProcessor
from database import DatabaseManager
from datetime import datetime, timedelta, date
import pandas as pd
import json
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database manager
db_manager = DatabaseManager()

class ResumeScreeningApp:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Get absolute paths
        base_dir = os.path.abspath(os.path.dirname(__file__))
        
        # PostgreSQL Configuration
        self.app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        self.app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-default-secret-key')
        
        # Initialize Flask-SQLAlchemy
        try:
            from models import db
            db.init_app(self.app)
            
            # Create tables if they don't exist
            with self.app.app_context():
                db.create_all()
                logger.info("Flask-SQLAlchemy tables created successfully")
        except Exception as e:
            logger.warning(f"Flask-SQLAlchemy initialization failed: {e}")
            logger.info("Continuing with custom DatabaseManager...")
        
        # Configuration
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        self.app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'temp_uploads')
        
        # Initialize resume processor
        self.resume_processor = ResumeProcessor()
        
        # Load ML models
        self._load_models()
        
        # Register routes and error handlers - ADD THESE LINES
        self._register_routes()
        self._register_error_handlers()
        
        # Add basic routes for testing - ADD THESE LINES
        @self.app.route('/')
        def home():
            return "Welcome to ResuAI! Your application is running correctly."
        
        @self.app.route('/routes')
        def list_routes():
            routes = []
            for rule in self.app.url_map.iter_rules():
                routes.append(f"{rule.endpoint}: {rule.rule}")
            return "<br>".join(routes)
    

    def _load_models(self):
        """Load pre-trained models"""
        try:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            
            # Load categorization models
            categorization_classifier_path = os.path.join(models_dir, 'rf_classifier_categorization.pkl')
            categorization_vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer_categorization.pkl')
            
            if os.path.exists(categorization_classifier_path):
                with open(categorization_classifier_path, 'rb') as f:
                    self.rf_classifier_categorization = pickle.load(f)
            else:
                self.rf_classifier_categorization = None
            
            if os.path.exists(categorization_vectorizer_path):
                with open(categorization_vectorizer_path, 'rb') as f:
                    self.tfidf_vectorizer_categorization = pickle.load(f)
            else:
                self.tfidf_vectorizer_categorization = None
            
            # Load job recommendation models
            recommendation_classifier_path = os.path.join(models_dir, 'rf_classifier_job_recommendation.pkl')
            recommendation_vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer_job_recommendation.pkl')
            
            if os.path.exists(recommendation_classifier_path):
                with open(recommendation_classifier_path, 'rb') as f:
                    self.rf_classifier_job_recommendation = pickle.load(f)
            else:
                self.rf_classifier_job_recommendation = None
            
            if os.path.exists(recommendation_vectorizer_path):
                with open(recommendation_vectorizer_path, 'rb') as f:
                    self.tfidf_vectorizer_job_recommendation = pickle.load(f)
            else:
                self.tfidf_vectorizer_job_recommendation = None
                
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            self.rf_classifier_categorization = None
            self.tfidf_vectorizer_categorization = None
            self.rf_classifier_job_recommendation = None
            self.tfidf_vectorizer_job_recommendation = None
    
    def _register_error_handlers(self):
        """Register error handlers"""
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'success': False, 'error': 'Resource not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    def _register_routes(self):
        """Register all application routes"""
        # Main routes
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/dashboard', 'dashboard', self.dashboard)
        self.app.add_url_rule('/dashboard/<path:section>', 'dashboard_section', self.dashboard)
        self.app.add_url_rule('/demo', 'demo', self.demo)
        
        # API routes
        self.app.add_url_rule('/api/health', 'health_check', self.health_check)
        self.app.add_url_rule('/api/upload', 'upload_resumes', self.upload_resumes, methods=['POST'])
        self.app.add_url_rule('/api/jobs', 'handle_jobs', self.handle_jobs, methods=['GET', 'POST'])
        self.app.add_url_rule('/api/jobs/<int:job_id>', 'handle_job', self.handle_job, methods=['GET', 'PUT', 'DELETE'])
        self.app.add_url_rule('/api/job-categories', 'handle_job_categories', self.handle_job_categories, methods=['GET', 'POST'])
        self.app.add_url_rule('/api/job-categories/<int:category_id>', 'handle_job_category', self.handle_job_category, methods=['PUT', 'DELETE'])
        self.app.add_url_rule('/api/candidates', 'get_candidates', self.get_candidates, methods=['GET'])
        self.app.add_url_rule('/api/candidates/<int:candidate_id>', 'handle_candidate', self.handle_candidate, methods=['GET', 'PUT', 'DELETE'])
        self.app.add_url_rule('/api/analytics', 'get_analytics', self.get_analytics, methods=['GET'])
        self.app.add_url_rule('/api/settings', 'handle_settings', self.handle_settings, methods=['GET', 'PUT'])
    
    def index(self):
        """Serve the landing page"""
        return render_template("index.html")
    
    def demo(self):
        """Serve the design demo page"""
        return render_template("demo.html")
    
    def dashboard(self, section=None):
        """Serve the dashboard page"""
        return render_template("dashboard.html")
    
    def health_check(self):
        """Health check endpoint"""
        return jsonify({"status": "healthy", "message": "Resume Screening AI is running"})
    
    def _is_allowed_file(self, filename):
        """Check if file type is allowed"""
        allowed_extensions = {'pdf', 'doc', 'docx', 'txt'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
    def upload_resumes(self):
        """Handle resume upload and processing"""
        try:
            if 'files[]' not in request.files:
                return jsonify({'success': False, 'error': 'No files uploaded'}), 400
            
            files = request.files.getlist('files[]')
            job_id = request.form.get('jobId')
            
            if not files or all(f.filename == '' for f in files):
                return jsonify({'success': False, 'error': 'No files selected'}), 400
            
            if not job_id:
                return jsonify({'success': False, 'error': 'Job ID is required'}), 400
            
            try:
                job_id = int(job_id)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid job ID'}), 400
            
            # Get job details
            job = db_manager.get_job(job_id)
            if not job:
                return jsonify({'success': False, 'error': 'Job not found'}), 404
            
            results = []
            errors = []
            
            for file in files:
                if file.filename != '' and self._is_allowed_file(file.filename):
                    try:
                        logger.info(f"Processing file: {file.filename}")
                        
                        # Extract text from resume
                        text = self.resume_processor.extract_text_from_file(file)
                        logger.info(f"Extracted text length: {len(text)} characters")
                        
                        if not text.strip():
                            errors.append(f"{file.filename}: No text could be extracted")
                            continue
                        
                        # Process resume and calculate score
                        result = self._process_resume_for_job(text, file.filename, job)
                        logger.info(f"Processing result for {file.filename}: {result}")
                        results.append(result)
                        
                        # Store candidate data
                        candidate_data = {
                            'name': result.get('name', 'Unknown'),
                            'email': result.get('email', ''),
                            'phone': result.get('phone', ''),
                            'resume_text': text,  # Add the resume text
                            'education': result.get('education', []),
                            'skills': ', '.join(result.get('allSkills', [])),  # Convert to comma-separated string
                            'job_id': job_id,
                            'category': result.get('predictedCategory', {}).get('category', 'Unknown'),
                            'score': result['matchScore'],
                            'status': 'pending'
                        }
                        
                        candidate_id = db_manager.create_candidate(candidate_data)
                        
                    except Exception as e:
                        error_msg = f"Error processing file {file.filename}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        continue
                else:
                    if file.filename:
                        errors.append(f"{file.filename}: Unsupported file type")
            
            response_data = {
                'success': True,
                'message': f'Successfully processed {len(results)} resumes',
                'results': results
            }
            
            if errors:
                response_data['warnings'] = errors
                response_data['message'] += f' ({len(errors)} files had errors)'
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error in upload_resumes: {e}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    def _process_resume_for_job(self, resume_text, filename, job):
        """Process a resume against a specific job"""
        # Extract basic information
        extracted_info = self.resume_processor.extract_basic_info(resume_text)
        
        # Extract education information
        education_info = self.resume_processor.extract_education(resume_text)
        
        # Predict job category using the ML model
        predicted_category = self._predict_job_category(resume_text)
        
        # Get job requirements
        job_requirements = [skill.strip().lower() for skill in job['requirements'].split(',')]
        
        # Extract skills from resume
        resume_skills = self.resume_processor.extract_skills(resume_text)
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        
        # Calculate match
        matched_skills = []
        missing_skills = []
        
        for req_skill in job_requirements:
            found = False
            for resume_skill in resume_skills_lower:
                if req_skill in resume_skill or resume_skill in req_skill:
                    matched_skills.append(req_skill.title())
                    found = True
                    break
            if not found:
                missing_skills.append(req_skill.title())
        
        # Calculate match score
        if job_requirements:
            match_score = (len(matched_skills) / len(job_requirements)) * 100
        else:
            match_score = 0
        
        return {
            'filename': filename,
            'name': extracted_info.get('name', 'Unknown'),
            'email': extracted_info.get('email', ''),
            'phone': extracted_info.get('phone', ''),
            'education': education_info,
            'allSkills': resume_skills,
            'predictedCategory': predicted_category,
            'matchScore': round(match_score),
            'matchedSkills': matched_skills,
            'missingSkills': missing_skills
        }
    
    def _predict_job_category(self, resume_text):
        """Predict job category for a resume using the ML model"""
        try:
            if self.rf_classifier_categorization and self.tfidf_vectorizer_categorization:
                # Transform the resume text using the TF-IDF vectorizer
                text_vectorized = self.tfidf_vectorizer_categorization.transform([resume_text])
                
                # Predict the category
                predicted_category = self.rf_classifier_categorization.predict(text_vectorized)[0]
                
                # Get prediction probability for confidence
                prediction_proba = self.rf_classifier_categorization.predict_proba(text_vectorized)[0]
                max_confidence = max(prediction_proba)
                
                return {
                    'category': predicted_category,
                    'confidence': round(max_confidence * 100, 2)
                }
            else:
                logger.warning("Categorization models not loaded")
                return {
                    'category': 'Unknown',
                    'confidence': 0
                }
        except Exception as e:
            logger.error(f"Error predicting job category: {e}")
            return {
                'category': 'Unknown',
                'confidence': 0
            }
    
    def handle_jobs(self):
        """Handle job listing and creation"""
        if request.method == 'GET':
            try:
                jobs = db_manager.get_all_jobs()
                logger.info(f"GET /api/jobs - Returning {len(jobs)} jobs")
                return jsonify({
                    'success': True,
                    'jobs': jobs
                })
            except Exception as e:
                logger.error(f"Error getting jobs: {e}")
                return jsonify({'success': False, 'error': 'Internal server error'}), 500
        
        elif request.method == 'POST':
            try:
                data = request.get_json()
                
                # Validate required fields
                required_fields = ['title', 'department', 'category', 'experience_level', 'description', 'requirements']
                for field in required_fields:
                    if not data.get(field):
                        return jsonify({'success': False, 'error': f'{field} is required'}), 400
                
                # Convert category name to category_id
                category_name = data['category']
                categories = db_manager.get_all_job_categories()
                category_id = None
                
                for category in categories:
                    if category['name'] == category_name:
                        category_id = category['id']
                        break
                
                if category_id is None:
                    return jsonify({'success': False, 'error': f'Category "{category_name}" not found'}), 400
                
                # Prepare job data with category_id
                job_data = {
                    'title': data['title'],
                    'department': data['department'],
                    'description': data['description'],
                    'requirements': data['requirements'],
                    'experience_level': data['experience_level'],
                    'category_id': category_id
                }
                
                # Create new job
                job_id = db_manager.create_job(job_data)
                job = db_manager.get_job(job_id)
                
                return jsonify({
                    'success': True,
                    'message': 'Job created successfully',
                    'job': job
                })
                
            except Exception as e:
                logger.error(f"Error creating job: {e}")
                return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    def handle_job(self, job_id):
        """Handle individual job operations"""
        if request.method == 'GET':
            try:
                job = db_manager.get_job(job_id)
                if not job:
                    return jsonify({'success': False, 'error': 'Job not found'}), 404
                return jsonify({'success': True, 'job': job})
            except Exception as e:
                logger.error(f"Error getting job: {e}")
                return jsonify({'success': False, 'error': 'Internal server error'}), 500
        
        elif request.method == 'PUT':
            try:
                data = request.get_json()
                
                # Convert category name to category_id if category is provided
                if 'category' in data:
                    category_name = data['category']
                    categories = db_manager.get_all_job_categories()
                    category_id = None
                    
                    for category in categories:
                        if category['name'] == category_name:
                            category_id = category['id']
                            break
                    
                    if category_id is None:
                        return jsonify({'success': False, 'error': f'Category "{category_name}" not found'}), 400
                    
                    # Replace category name with category_id
                    data = data.copy()  # Don't modify original data
                    del data['category']
                    data['category_id'] = category_id
                
                success = db_manager.update_job(job_id, data)
                
                if not success:
                    return jsonify({'success': False, 'error': 'Job not found'}), 404
                
                job = db_manager.get_job(job_id)
                return jsonify({
                    'success': True,
                    'message': 'Job updated successfully',
                    'job': job
                })
                
            except Exception as e:
                logger.error(f"Error updating job: {e}")
                return jsonify({'success': False, 'error': 'Internal server error'}), 500
        
        elif request.method == 'DELETE':
            try:
                success = db_manager.delete_job(job_id)
                if not success:
                    return jsonify({'success': False, 'error': 'Job not found'}), 404
                
                return jsonify({'success': True, 'message': 'Job deleted successfully'})
            except Exception as e:
                logger.error(f"Error deleting job: {e}")
                return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    def handle_job_categories(self):
        """Handle job category listing and creation"""
        if request.method == 'GET':
            try:
                categories = db_manager.get_all_job_categories()
                logger.info(f"GET /api/job-categories - Returning {len(categories)} categories")
                return jsonify({
                    'success': True,
                    'categories': categories
                })
            except Exception as e:
                logger.error(f"Error getting job categories: {e}")
                return jsonify({'success': False, 'error': 'Internal server error'}), 500
        
        elif request.method == 'POST':
            try:
                data = request.get_json()
                
                if not data.get('name'):
                    return jsonify({'success': False, 'error': 'Category name is required'}), 400
                
                # Check if category already exists
                existing_categories = db_manager.get_all_job_categories()
                for category in existing_categories:
                    if category['name'].lower() == data['name'].lower():
                        return jsonify({'success': False, 'error': 'Category already exists'}), 400
                
                # Create new category
                category_id = db_manager.create_job_category(
                    data['name'], 
                    data.get('description', '')
                )
                
                category = {
                    'id': category_id,
                    'name': data['name'],
                    'description': data.get('description', ''),
                    'created_at': datetime.now().isoformat()
                }
                
                return jsonify({
                    'success': True,
                    'message': 'Category created successfully',
                    'category': category
                })
                
            except Exception as e:
                logger.error(f"Error creating category: {e}")
                return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    def handle_job_category(self, category_id):
        """Handle individual job category operations"""
        if request.method == 'PUT':
            try:
                data = request.get_json()
                
                name = data.get('name')
                description = data.get('description')
                
                success = db_manager.update_job_category(category_id, name, description)
                if not success:
                    return jsonify({'success': False, 'error': 'Category not found'}), 404
                
                # Get updated category (simplified response)
                category = {
                    'id': category_id,
                    'name': name if name else 'Updated Category',
                    'description': description if description else ''
                }
                
                return jsonify({
                    'success': True,
                    'message': 'Category updated successfully',
                    'category': category
                })
                
            except Exception as e:
                logger.error(f"Error updating category: {e}")
                return jsonify({'success': False, 'error': 'Internal server error'}), 500
        
        elif request.method == 'DELETE':
            try:
                # Get category name before deleting to check usage
                categories = db_manager.get_all_job_categories()
                category_name = None
                for cat in categories:
                    if cat['id'] == category_id:
                        category_name = cat['name']
                        break
                
                if not category_name:
                    return jsonify({'success': False, 'error': 'Category not found'}), 404
                
                # Check if any jobs use this category
                jobs_count = db_manager.check_category_in_use(category_name)
                if jobs_count > 0:
                    return jsonify({
                        'success': False, 
                        'error': f'Cannot delete category. {jobs_count} jobs are using this category.'
                    }), 400
                
                success = db_manager.delete_job_category(category_id)
                if not success:
                    return jsonify({'success': False, 'error': 'Category not found'}), 404
                
                return jsonify({'success': True, 'message': 'Category deleted successfully'})
            except Exception as e:
                logger.error(f"Error deleting category: {e}")
                return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    def get_candidates(self):
        """Get list of candidates organized by job categories"""
        try:
            candidates = db_manager.get_all_candidates()
            jobs = db_manager.get_all_jobs()
            
            # Group candidates by job
            candidates_by_job = {}
            
            # Initialize with all jobs
            for job in jobs:
                candidates_by_job[job['id']] = {
                    'job_title': job['title'],
                    'job_category': job['category'],
                    'job_description': job['description'],
                    'job_requirements': job['requirements'],
                    'candidates': []
                }
            
            # Add candidates to their respective jobs
            for candidate in candidates:
                job_id = candidate.get('job_id')
                if job_id and job_id in candidates_by_job:
                    # Format education as string for display
                    education_str = ""
                    if candidate.get('education'):
                        education_items = []
                        for edu in candidate['education']:
                            if isinstance(edu, dict):
                                degree = edu.get('degree', '')
                                year = edu.get('year', '')
                                if degree:
                                    education_items.append(f"{degree} ({year})" if year else degree)
                        education_str = ", ".join(education_items) if education_items else "Not specified"
                    else:
                        education_str = "Not specified"
                    
                    # Format skills as string
                    skills_list = candidate.get('skills', [])
                    skills_str = ", ".join(skills_list[:10]) + ("..." if len(skills_list) > 10 else "") if skills_list else "Not specified"
                    
                    # Format predicted category - handle if it's stored as a simple string
                    predicted_category_str = candidate.get('category', 'Unknown')
                    
                    formatted_candidate = {
                        'id': candidate['id'],
                        'name': candidate['name'],
                        'email': candidate['email'],
                        'phone': candidate['phone'],
                        'education': education_str,
                        'skills': skills_str,
                        'all_skills': skills_list,
                        'predicted_category': predicted_category_str,
                        'score': candidate['score'],
                        'status': candidate['status'],
                        'created_at': candidate['created_at'].isoformat() if candidate.get('created_at') else None,
                        'updated_at': candidate['updated_at'].isoformat() if candidate.get('updated_at') else None
                    }
                    
                    candidates_by_job[job_id]['candidates'].append(formatted_candidate)
            
            return jsonify({
                'success': True,
                'candidates_by_job': candidates_by_job,
                'total_candidates': len(candidates)
            })
            
        except Exception as e:
            logger.error(f"Error getting candidates: {e}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    def handle_candidate(self, candidate_id):
        """Handle individual candidate operations"""
        if request.method == 'GET':
            try:
                candidate = db_manager.get_candidate(candidate_id)
                if not candidate:
                    return jsonify({'success': False, 'error': 'Candidate not found'}), 404
                
                # Format detailed candidate info
                detailed_candidate = {
                    'id': candidate['id'],
                    'name': candidate['name'],
                    'email': candidate['email'],
                    'phone': candidate['phone'],
                    'matchScore': candidate['score'],
                    'status': candidate['status'],
                    'category': candidate['category'],
                    'job_title': candidate['job_title'],
                    'skills': candidate.get('all_skills', []),
                    'education': candidate.get('education', []),
                    'matched_skills': candidate.get('matched_skills', []),
                    'missing_skills': candidate.get('missing_skills', []),
                    'predicted_category': candidate.get('predicted_category', {}),
                    'filename': candidate.get('filename', ''),
                    'updated_at': candidate['updated_at']
                }
                
                return jsonify({'success': True, 'candidate': detailed_candidate})
            except Exception as e:
                logger.error(f"Error getting candidate: {e}")
                return jsonify({'success': False, 'error': 'Internal server error'}), 500
        
        elif request.method == 'PUT':
            try:
                data = request.get_json()
                
                if 'status' in data:
                    success = db_manager.update_candidate(candidate_id, {'status': data['status']})
                    if not success:
                        return jsonify({'success': False, 'error': 'Candidate not found'}), 404
                    
                    candidate = db_manager.get_candidate(candidate_id)
                    return jsonify({
                        'success': True,
                        'message': 'Candidate updated successfully',
                        'candidate': candidate
                    })
                else:
                    return jsonify({'success': False, 'error': 'No valid fields to update'}), 400
                
            except Exception as e:
                logger.error(f"Error updating candidate: {e}")
                return jsonify({'success': False, 'error': 'Internal server error'}), 500
        
        elif request.method == 'DELETE':
            try:
                success = db_manager.delete_candidate(candidate_id)
                if not success:
                    return jsonify({'success': False, 'error': 'Candidate not found'}), 404
                
                return jsonify({'success': True, 'message': 'Candidate removed successfully'})
            except Exception as e:
                logger.error(f"Error deleting candidate: {e}")
                return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    def get_analytics(self):
        """Get analytics data"""
        try:
            # Get summary data from database
            summary = db_manager.get_analytics_summary()
            
            # Create more realistic daily stats based on actual data
            today = datetime.now().date()
            daily_stats = []
            
            # Use actual data for recent days and reduce for older days
            for i in range(30):
                date_obj = today - timedelta(days=i)
                
                # Simulate realistic data degradation over time
                time_factor = max(0.1, 1 - (i * 0.1))  # Reduce by 10% each day going back
                
                daily_stats.append({
                    'date': date_obj.strftime('%Y-%m-%d'),
                    'total_resumes': max(0, int(summary['total_resumes'] * time_factor)),
                    'processed_resumes': max(0, int(summary['processed_resumes'] * time_factor)),
                    'shortlisted': max(0, int(summary['shortlisted'] * time_factor)),
                    'rejected': max(0, int(summary['rejected'] * time_factor)),
                    'job_category_stats': json.dumps(summary['job_category_stats'])
                })
            
            daily_stats.reverse()  # Show oldest to newest
            
            return jsonify({
                'success': True,
                'summary': {
                    'total_resumes': summary['total_resumes'],
                    'processed_resumes': summary['processed_resumes'],
                    'shortlisted': summary['shortlisted'],
                    'rejected': summary['rejected'],
                    'avg_score': summary['avg_score'],
                    'avg_processing_time': 3  # Fixed value for now
                },
                'daily_stats': daily_stats,
                'job_category_stats': summary['job_category_stats']
            })
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    
    def handle_settings(self):
        """Handle application settings"""
        if request.method == 'GET':
            try:
                settings = db_manager.get_all_settings()
                return jsonify({
                    'success': True,
                    'settings': settings
                })
            except Exception as e:
                logger.error(f"Error getting settings: {e}")
                return jsonify({'success': False, 'error': 'Internal server error'}), 500
        
        elif request.method == 'PUT':
            try:
                data = request.get_json()
                db_manager.update_settings(data)
                
                settings = db_manager.get_all_settings()
                return jsonify({
                    'success': True,
                    'message': 'Settings updated successfully',
                    'settings': settings
                })
                
            except Exception as e:
                logger.error(f"Error updating settings: {e}")
                return jsonify({'success': False, 'error': 'Internal server error'}), 500

def create_app():
    """Create and configure the Flask application"""
    app_instance = ResumeScreeningApp()
    return app_instance.app

if __name__ == '__main__':
    try:
        logger.info("Starting Resume Screening Application...")
        
        # Initialize database
        jobs = db_manager.get_all_jobs()
        categories = db_manager.get_all_job_categories()
        logger.info(f"Database initialized with {len(jobs)} jobs and {len(categories)} categories")
        
        app = create_app()
        logger.info("Flask app created successfully")
        logger.info("Starting server on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
