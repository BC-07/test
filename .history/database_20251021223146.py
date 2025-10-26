import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
from dotenv import load_dotenv
import bcrypt

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.getenv('DATABASE_URL')
        self.use_sqlite = False  # Flag to indicate SQLite fallback
        
        if not self.db_url:
            # Fallback to individual components
            db_user = os.getenv('DB_USER')
            db_password = os.getenv('DB_PASSWORD')
            db_host = os.getenv('DB_HOST', 'localhost')
            db_port = os.getenv('DB_PORT', '5432')
            db_name = os.getenv('DB_NAME')
            
            if not all([db_user, db_password, db_name]):
                # No PostgreSQL config, check for SQLite database
                sqlite_path = os.path.join(os.path.dirname(__file__), 'resume_screening.db')
                if os.path.exists(sqlite_path):
                    logger.info("PostgreSQL not configured, using SQLite database")
                    self.use_sqlite = True
                    self._init_sqlite_assessment()
                    self.init_database()
                    return
                else:
                    raise ValueError("Database configuration missing. Please set DATABASE_URL or individual DB_* environment variables, or ensure resume_screening.db exists.")
            
            self.db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        self.init_database()
    
    def _init_sqlite_assessment(self):
        """Initialize SQLite assessment database helper"""
        try:
            from sqlite_assessment_db import SQLiteAssessmentDB
            self.sqlite_assessment = SQLiteAssessmentDB()
            logger.info("SQLite assessment database helper initialized")
        except ImportError as e:
            logger.error(f"Could not import SQLite assessment helper: {e}")
            self.sqlite_assessment = None
    
    def get_connection(self):
        """Get database connection"""
        if self.use_sqlite:
            # This should not be used for SQLite, but provide a graceful error
            raise RuntimeError("SQLite mode active - use sqlite_assessment methods")
        
        try:
            conn = psycopg2.connect(
                self.db_url,
                cursor_factory=RealDictCursor
            )
            return conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def init_database(self):
        """Initialize database with all required tables"""
        if self.use_sqlite:
            # SQLite database already initialized by migration
            logger.info("Using existing SQLite database")
            return
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Read and execute schema
                    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
                    with open(schema_path, 'r') as f:
                        cursor.execute(f.read())
                    conn.commit()
                    logger.info("Database initialized successfully")
        except FileNotFoundError:
            logger.error("schema.sql file not found")
            raise
        except psycopg2.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    # Job Category Operations
    def create_job_category(self, name: str, description: str = "") -> int:
        """Create a new job category"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO job_categories (name, description)
                    VALUES (%s, %s)
                    RETURNING id
                ''', (name, description))
                result = cursor.fetchone()
                conn.commit()
                return result['id']
    
    def get_all_job_categories(self) -> List[Dict]:
        """Get all job categories"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM job_categories ORDER BY name")
                return [dict(row) for row in cursor.fetchall()]
    
    def update_job_category(self, category_id: int, name: str = None, description: str = None) -> bool:
        """Update a job category"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                fields = []
                values = []
                
                if name is not None:
                    fields.append("name = %s")
                    values.append(name)
                if description is not None:
                    fields.append("description = %s")
                    values.append(description)
                
                if not fields:
                    return False
                
                fields.append("updated_at = CURRENT_TIMESTAMP")
                values.append(category_id)
                
                query = f"UPDATE job_categories SET {', '.join(fields)} WHERE id = %s"
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
    
    def delete_job_category(self, category_id: int) -> bool:
        """Delete a job category"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM job_categories WHERE id = %s", (category_id,))
                conn.commit()
                return cursor.rowcount > 0
    
    def check_category_in_use(self, category_name: str) -> int:
        """Check how many jobs use a specific category"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT COUNT(*) as job_count
                    FROM jobs j
                    JOIN job_categories jc ON j.category_id = jc.id
                    WHERE jc.name = %s
                ''', (category_name,))
                result = cursor.fetchone()
                return result['job_count'] if result else 0

    # Job Operations
    def create_job(self, job_data: Dict) -> int:
        """Create a new job and return its ID"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO jobs (title, department, description, requirements, experience_level, category_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (
                    job_data['title'],
                    job_data['department'],
                    job_data['description'],
                    job_data['requirements'],
                    job_data['experience_level'],
                    job_data['category_id']
                ))
                result = cursor.fetchone()
                conn.commit()
                return result['id']
    
    def get_job(self, job_id: int) -> Optional[Dict]:
        """Get a job by ID"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT j.*, COALESCE(jc.name, 'Uncategorized') as category_name 
                    FROM jobs j
                    LEFT JOIN job_categories jc ON j.category_id = jc.id
                    WHERE j.id = %s
                ''', (job_id,))
                row = cursor.fetchone()
                if row:
                    job = dict(row)
                    # Add category field for backward compatibility
                    job['category'] = job['category_name']
                    return job
                return None
    
    def get_all_jobs(self) -> List[Dict]:
        """Get all jobs"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT j.*, COALESCE(jc.name, 'Uncategorized') as category_name 
                    FROM jobs j
                    LEFT JOIN job_categories jc ON j.category_id = jc.id
                    ORDER BY j.created_at DESC
                ''')
                jobs = []
                for row in cursor.fetchall():
                    job = dict(row)
                    # Add category field for backward compatibility
                    job['category'] = job['category_name']
                    jobs.append(job)
                return jobs
    
    def update_job(self, job_id: int, job_data: Dict) -> bool:
        """Update a job"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                fields = []
                values = []
                
                for key, value in job_data.items():
                    if key in ['title', 'department', 'description', 'requirements', 'experience_level', 'category_id', 'status']:
                        fields.append(f"{key} = %s")
                        values.append(value)
                
                if not fields:
                    return False
                
                fields.append("updated_at = CURRENT_TIMESTAMP")
                values.append(job_id)
                
                query = f"UPDATE jobs SET {', '.join(fields)} WHERE id = %s"
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
    
    def delete_job(self, job_id: int) -> bool:
        """Delete a job"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
                conn.commit()
                return cursor.rowcount > 0
    
    # Candidate Operations
    def create_candidate(self, candidate_data: Dict) -> int:
        """Create a new candidate"""
        if self.use_sqlite:
            # SQLite implementation
            import sqlite3
            sqlite_path = os.path.join(os.path.dirname(__file__), 'resume_screening.db')
            
            with sqlite3.connect(sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO candidates 
                    (name, email, phone, linkedin, github, resume_text, category, skills, 
                     education, experience, status, score, job_id, notes, processing_type,
                     pds_data, eligibility, training, volunteer_work, personal_references, 
                     government_ids, ocr_confidence,
                     pds_extracted_data, total_education_entries, total_work_positions,
                     extraction_status, uploaded_filename, latest_total_score,
                     latest_percentage_score, latest_recommendation, upload_batch_id,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            datetime('now'), datetime('now'))
                ''', (
                    candidate_data.get('name'),
                    candidate_data.get('email'),
                    candidate_data.get('phone'),
                    candidate_data.get('linkedin'),
                    candidate_data.get('github'),
                    candidate_data['resume_text'],
                    candidate_data.get('category'),
                    json.dumps(candidate_data.get('skills', [])),
                    json.dumps(candidate_data.get('education', [])),
                    json.dumps(candidate_data.get('experience', [])),
                    candidate_data.get('status', 'new'),
                    candidate_data.get('score', 0.0),
                    candidate_data.get('job_id'),
                    candidate_data.get('notes'),
                    candidate_data.get('processing_type', 'resume'),
                    # PDS-specific fields
                    json.dumps(candidate_data.get('pds_data')) if candidate_data.get('pds_data') else None,
                    json.dumps(candidate_data.get('eligibility', [])),
                    json.dumps(candidate_data.get('training', [])),
                    json.dumps(candidate_data.get('volunteer_work', [])),
                    json.dumps(candidate_data.get('personal_references', [])),
                    json.dumps(candidate_data.get('government_ids', {})),
                    candidate_data.get('ocr_confidence'),
                    json.dumps(candidate_data.get('pds_extracted_data')) if candidate_data.get('pds_extracted_data') else None,
                    candidate_data.get('total_education_entries', 0),
                    candidate_data.get('total_work_positions', 0),
                    candidate_data.get('extraction_status', 'completed'),
                    candidate_data.get('uploaded_filename'),
                    candidate_data.get('latest_total_score'),
                    candidate_data.get('latest_percentage_score'),
                    candidate_data.get('latest_recommendation'),
                    candidate_data.get('upload_batch_id')
                ))
                return cursor.lastrowid
        else:
            # PostgreSQL implementation
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        INSERT INTO candidates 
                        (name, email, phone, linkedin, github, resume_text, category, skills, 
                         education, experience, status, score, job_id, notes, processing_type,
                         pds_data, eligibility, training, volunteer_work, personal_references, 
                         government_ids, ocr_confidence,
                         pds_extracted_data, total_education_entries, total_work_positions,
                         extraction_status, uploaded_filename, latest_total_score,
                         latest_percentage_score, latest_recommendation, upload_batch_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    ''', (
                        candidate_data.get('name'),
                        candidate_data.get('email'),
                        candidate_data.get('phone'),
                        candidate_data.get('linkedin'),
                        candidate_data.get('github'),
                        candidate_data['resume_text'],
                        candidate_data.get('category'),
                        candidate_data.get('skills'),
                        json.dumps(candidate_data.get('education', [])),
                        json.dumps(candidate_data.get('experience', [])),
                        candidate_data.get('status', 'new'),
                        candidate_data.get('score', 0.0),
                        candidate_data.get('job_id'),
                        candidate_data.get('notes'),
                        candidate_data.get('processing_type', 'resume'),
                        # PDS-specific fields that exist in PostgreSQL
                        json.dumps(candidate_data.get('pds_data')) if candidate_data.get('pds_data') else None,
                        json.dumps(candidate_data.get('eligibility', [])),
                        json.dumps(candidate_data.get('training', [])),
                        json.dumps(candidate_data.get('volunteer_work', [])),
                        json.dumps(candidate_data.get('personal_references', [])),
                        json.dumps(candidate_data.get('government_ids', {})),
                        candidate_data.get('ocr_confidence'),
                        json.dumps(candidate_data.get('pds_extracted_data')) if candidate_data.get('pds_extracted_data') else None,
                        candidate_data.get('total_education_entries', 0),
                        candidate_data.get('total_work_positions', 0),
                        candidate_data.get('extraction_status', 'completed'),
                        candidate_data.get('uploaded_filename'),
                        candidate_data.get('latest_total_score'),
                        candidate_data.get('latest_percentage_score'),
                        candidate_data.get('latest_recommendation'),
                        candidate_data.get('upload_batch_id')
                    ))
                    result = cursor.fetchone()
                    return result['id']
    
    def get_candidate(self, candidate_id: int) -> Optional[Dict]:
        """Get a candidate by ID"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT c.*, j.title as job_title 
                    FROM candidates c
                    LEFT JOIN jobs j ON c.job_id = j.id
                    WHERE c.id = %s
                ''', (candidate_id,))
                row = cursor.fetchone()
                if row:
                    candidate = dict(row)
                    # Parse JSON fields safely
                    json_fields = ['education', 'experience', 'pds_data', 'eligibility', 
                                 'training', 'volunteer_work', 'personal_references', 'government_ids']
                    
                    for field in json_fields:
                        try:
                            if isinstance(candidate.get(field), str):
                                candidate[field] = json.loads(candidate[field])
                            elif candidate.get(field) is None:
                                candidate[field] = [] if field != 'government_ids' and field != 'pds_data' else {}
                        except (json.JSONDecodeError, TypeError):
                            candidate[field] = [] if field != 'government_ids' and field != 'pds_data' else {}
                    
                    candidate['skills'] = candidate['skills'].split(',') if candidate['skills'] else []
                    return candidate
                return None
    
    def get_all_candidates(self) -> List[Dict]:
        """Get all candidates"""
        if self.use_sqlite:
            # SQLite implementation
            import sqlite3
            sqlite_path = os.path.join(os.path.dirname(__file__), 'resume_screening.db')
            
            with sqlite3.connect(sqlite_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM candidates
                    ORDER BY score DESC
                ''')
                candidates = []
                for row in cursor.fetchall():
                    candidate = dict(row)
                    # Parse JSON fields safely
                    json_fields = ['education', 'experience', 'pds_data', 'eligibility', 
                                 'training', 'volunteer_work', 'personal_references', 'government_ids']
                    
                    for field in json_fields:
                        try:
                            if isinstance(candidate.get(field), str):
                                candidate[field] = json.loads(candidate[field])
                            elif candidate.get(field) is None:
                                candidate[field] = [] if field != 'government_ids' and field != 'pds_data' else {}
                        except (json.JSONDecodeError, TypeError):
                            candidate[field] = [] if field != 'government_ids' and field != 'pds_data' else {}
                    
                    # Handle skills field
                    if candidate.get('skills'):
                        if isinstance(candidate['skills'], str):
                            try:
                                candidate['skills'] = json.loads(candidate['skills'])
                            except json.JSONDecodeError:
                                candidate['skills'] = candidate['skills'].split(',') if candidate['skills'] else []
                    else:
                        candidate['skills'] = []
                    
                    candidates.append(candidate)
                return candidates
        else:
            # PostgreSQL implementation
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        SELECT c.*, j.title as job_title 
                        FROM candidates c
                        LEFT JOIN jobs j ON c.job_id = j.id
                        ORDER BY c.score DESC
                    ''')
                    candidates = []
                    for row in cursor.fetchall():
                        candidate = dict(row)
                        # Parse JSON fields safely
                        json_fields = ['education', 'experience', 'pds_data', 'eligibility', 
                                     'training', 'volunteer_work', 'personal_references', 'government_ids']
                        
                        for field in json_fields:
                            try:
                                if isinstance(candidate.get(field), str):
                                    candidate[field] = json.loads(candidate[field])
                                elif candidate.get(field) is None:
                                    candidate[field] = [] if field != 'government_ids' and field != 'pds_data' else {}
                            except (json.JSONDecodeError, TypeError):
                                candidate[field] = [] if field != 'government_ids' and field != 'pds_data' else {}
                        
                        candidate['skills'] = candidate['skills'].split(',') if candidate['skills'] else []
                        candidates.append(candidate)
                    return candidates
    
    def update_candidate(self, candidate_id: int, candidate_data: Dict) -> bool:
        """Update a candidate"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                fields = []
                values = []
                
                # Standard fields
                standard_fields = [
                    'name', 'email', 'phone', 'linkedin', 'github', 'status', 'score', 'job_id', 'notes',
                    'latest_total_score', 'latest_percentage_score', 'latest_recommendation',
                    'processing_type', 'extraction_status', 'uploaded_filename', 'upload_batch_id',
                    'total_education_entries', 'total_work_positions'
                ]
                
                # JSON fields
                json_fields = ['education', 'experience', 'pds_extracted_data']
                
                for key, value in candidate_data.items():
                    if key in standard_fields:
                        fields.append(f"{key} = %s")
                        values.append(value)
                    elif key in json_fields:
                        fields.append(f"{key} = %s")
                        values.append(json.dumps(value) if value is not None else None)
                
                if not fields:
                    return False
                
                fields.append("updated_at = CURRENT_TIMESTAMP")
                values.append(candidate_id)
                
                query = f"UPDATE candidates SET {', '.join(fields)} WHERE id = %s"
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
    
    def update_candidate_potential_score(self, candidate_id: int, potential_score: float) -> bool:
        """Update potential score for a candidate"""
        try:
            logger.info(f"Updating potential score for candidate {candidate_id} to {potential_score}")
            logger.info(f"Using database mode: {'SQLite' if self.use_sqlite else 'PostgreSQL'}")
            
            if self.use_sqlite:
                # SQLite implementation
                import sqlite3
                sqlite_path = os.path.join(os.path.dirname(__file__), 'resume_screening.db')
                logger.info(f"SQLite path: {sqlite_path}")
                conn = sqlite3.connect(sqlite_path)
                cursor = conn.cursor()
                
                # Update potential score
                cursor.execute(
                    'UPDATE candidates SET potential_score = ? WHERE id = ?',
                    (potential_score, candidate_id)
                )
                
                conn.commit()
                rows_affected = cursor.rowcount
                conn.close()
                logger.info(f"SQLite update completed. Rows affected: {rows_affected}")
                return rows_affected > 0
            else:
                # PostgreSQL implementation
                logger.info("Using PostgreSQL for potential score update")
                with self.get_connection() as conn:
                    with conn.cursor() as cursor:
                        # Update potential score
                        cursor.execute(
                            "UPDATE candidates SET potential_score = %s WHERE id = %s",
                            (potential_score, candidate_id)
                        )
                        conn.commit()
                        rows_affected = cursor.rowcount
                        logger.info(f"PostgreSQL update completed. Rows affected: {rows_affected}")
                        return rows_affected > 0
        except Exception as e:
            logger.error(f"Error updating candidate potential score: {e}")
            return False
    
    def get_candidates_by_batch(self, batch_id: str) -> List[Dict]:
        """Get all candidates from a specific batch"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM candidates 
                    WHERE upload_batch_id = %s 
                    ORDER BY created_at DESC
                """, (batch_id,))
                
                candidates = []
                for row in cursor.fetchall():
                    candidate = dict(row)
                    # Parse JSON fields
                    if candidate.get('education'):
                        try:
                            candidate['education'] = json.loads(candidate['education'])
                        except:
                            candidate['education'] = []
                    
                    if candidate.get('experience'):
                        try:
                            candidate['experience'] = json.loads(candidate['experience'])
                        except:
                            candidate['experience'] = []
                    
                    if candidate.get('pds_extracted_data'):
                        try:
                            candidate['pds_extracted_data'] = json.loads(candidate['pds_extracted_data'])
                        except:
                            candidate['pds_extracted_data'] = {}
                    
                    candidates.append(candidate)
                return candidates
    
    def delete_candidate(self, candidate_id: int) -> bool:
        """Delete a candidate"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM candidates WHERE id = %s", (candidate_id,))
                conn.commit()
                return cursor.rowcount > 0
    
    # Analytics Operations
    def get_analytics_summary(self) -> Dict:
        """Get analytics summary data"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Basic counts
                cursor.execute("SELECT COUNT(*) as total_resumes FROM candidates")
                total_resumes = cursor.fetchone()['total_resumes']
                
                cursor.execute("SELECT COUNT(*) as processed FROM candidates WHERE status != 'new'")
                processed_resumes = cursor.fetchone()['processed']
                
                cursor.execute("SELECT COUNT(*) as shortlisted FROM candidates WHERE status = 'shortlisted'")
                shortlisted = cursor.fetchone()['shortlisted']
                
                cursor.execute("SELECT COUNT(*) as rejected FROM candidates WHERE status = 'rejected'")
                rejected = cursor.fetchone()['rejected']
                
                # PDS specific counts
                cursor.execute("SELECT COUNT(*) as total_pds FROM candidates WHERE processing_type = 'ocr' OR processing_type = 'pds'")
                total_pds = cursor.fetchone()['total_pds']
                
                cursor.execute("SELECT COUNT(*) as processed_pds FROM candidates WHERE (processing_type = 'ocr' OR processing_type = 'pds') AND status != 'new'")
                processed_pds = cursor.fetchone()['processed_pds']
                
                # Processing type breakdown
                cursor.execute("SELECT processing_type, COUNT(*) as count FROM candidates WHERE processing_type IS NOT NULL GROUP BY processing_type")
                processing_type_stats = {row['processing_type']: row['count'] for row in cursor.fetchall()}
                
                # Average score
                cursor.execute("SELECT AVG(score) as avg_score FROM candidates")
                avg_score_result = cursor.fetchone()['avg_score']
                avg_score = float(avg_score_result) if avg_score_result else 0.0
                
                # Job category stats
                cursor.execute("SELECT category, COUNT(*) as count FROM candidates WHERE category IS NOT NULL GROUP BY category")
                job_category_stats = {row['category']: row['count'] for row in cursor.fetchall()}
                
                return {
                    'total_resumes': total_resumes,
                    'processed_resumes': processed_resumes,
                    'total_pds': total_pds,
                    'processed_pds': processed_pds,
                    'shortlisted': shortlisted,
                    'rejected': rejected,
                    'avg_score': round(avg_score, 2),
                    'processing_type_stats': processing_type_stats,
                    'job_category_stats': job_category_stats
                }
    
    # Settings Operations
    def get_setting(self, key: str) -> Optional[str]:
        """Get a setting value"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT value FROM settings WHERE key = %s", (key,))
                row = cursor.fetchone()
                return row['value'] if row else None
    
    def set_setting(self, key: str, value: str):
        """Set a setting value"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO settings (key, value, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (key) DO UPDATE SET 
                    value = EXCLUDED.value,
                    updated_at = CURRENT_TIMESTAMP
                ''', (key, value))
                conn.commit()
    
    def get_all_settings(self) -> Dict:
        """Get all settings"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT key, value FROM settings")
                return {row['key']: row['value'] for row in cursor.fetchall()}
    
    def update_settings(self, settings: Dict):
        """Update multiple settings"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                for key, value in settings.items():
                    cursor.execute('''
                        INSERT INTO settings (key, value, updated_at)
                        VALUES (%s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (key) DO UPDATE SET 
                        value = EXCLUDED.value,
                        updated_at = CURRENT_TIMESTAMP
                    ''', (key, str(value)))
                conn.commit()
    
    # User Authentication Operations
    def create_user(self, email: str, password: str, first_name: str, last_name: str, is_admin: bool = False) -> int:
        """Create a new user with hashed password"""
        import bcrypt
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO users (email, password_hash, first_name, last_name, is_admin)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                ''', (email, password_hash, first_name, last_name, is_admin))
                result = cursor.fetchone()
                conn.commit()
                return result['id']
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user data if valid"""
        import bcrypt
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT id, email, password_hash, first_name, last_name, is_admin, is_active, last_login
                    FROM users 
                    WHERE email = %s AND is_active = TRUE
                ''', (email,))
                user = cursor.fetchone()
                
                if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                    # Update last login
                    cursor.execute('''
                        UPDATE users SET last_login = CURRENT_TIMESTAMP 
                        WHERE id = %s
                    ''', (user['id'],))
                    conn.commit()
                    return dict(user)
                return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT id, email, first_name, last_name, is_admin, is_active, last_login, created_at
                    FROM users 
                    WHERE id = %s
                ''', (user_id,))
                user = cursor.fetchone()
                return dict(user) if user else None
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT id, email, first_name, last_name, is_admin, is_active, last_login, created_at
                    FROM users 
                    WHERE email = %s
                ''', (email,))
                user = cursor.fetchone()
                return dict(user) if user else None
    
    def get_all_users(self) -> List[Dict]:
        """Get all users"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT id, email, first_name, last_name, is_admin, is_active, last_login, created_at
                    FROM users 
                    ORDER BY created_at DESC
                ''')
                return [dict(row) for row in cursor.fetchall()]
    
    def update_user(self, user_id: int, **kwargs) -> bool:
        """Update user information"""
        import bcrypt
        allowed_fields = ['email', 'first_name', 'last_name', 'is_admin', 'is_active', 'password']
        fields = []
        values = []
        
        for key, value in kwargs.items():
            if key in allowed_fields:
                if key == 'password':
                    password_hash = bcrypt.hashpw(value.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                    fields.append("password_hash = %s")
                    values.append(password_hash)
                else:
                    fields.append(f"{key} = %s")
                    values.append(value)
        
        if not fields:
            return False
        
        fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(user_id)
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f'''
                    UPDATE users SET {", ".join(fields)}
                    WHERE id = %s
                ''', values)
                conn.commit()
                return cursor.rowcount > 0
    
    def delete_user(self, user_id: int) -> bool:
        """Delete user (soft delete by setting inactive)"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    UPDATE users SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                ''', (user_id,))
                conn.commit()
                return cursor.rowcount > 0
    
    # PDS Candidates Operations
    def create_pds_candidate(self, pds_data: Dict) -> int:
        """Create a new PDS candidate"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO pds_candidates 
                    (name, email, phone, job_id, score, status, filename, file_size, 
                     personal_info, family_background, educational_background, 
                     civil_service_eligibility, work_experience, voluntary_work, 
                     learning_development, other_information, personal_references, 
                     government_ids, highest_education, years_of_experience, 
                     government_service_years, civil_service_eligible, scoring_breakdown, 
                     matched_qualifications, areas_for_improvement, extraction_success, 
                     extraction_errors, processing_notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (
                    pds_data.get('name'),
                    pds_data.get('email'),
                    pds_data.get('phone'),
                    pds_data.get('job_id'),
                    pds_data.get('score', 0.0),
                    pds_data.get('status', 'new'),
                    pds_data.get('filename'),
                    pds_data.get('file_size'),
                    json.dumps(pds_data.get('personal_info', {})),
                    json.dumps(pds_data.get('family_background', {})),
                    json.dumps(pds_data.get('educational_background', {})),
                    json.dumps(pds_data.get('civil_service_eligibility', [])),
                    json.dumps(pds_data.get('work_experience', [])),
                    json.dumps(pds_data.get('voluntary_work', [])),
                    json.dumps(pds_data.get('learning_development', [])),
                    json.dumps(pds_data.get('other_information', {})),
                    json.dumps(pds_data.get('personal_references', [])),
                    json.dumps(pds_data.get('government_ids', {})),
                    pds_data.get('highest_education'),
                    pds_data.get('years_of_experience', 0),
                    pds_data.get('government_service_years', 0),
                    pds_data.get('civil_service_eligible', False),
                    json.dumps(pds_data.get('scoring_breakdown', {})),
                    json.dumps(pds_data.get('matched_qualifications', [])),
                    json.dumps(pds_data.get('areas_for_improvement', [])),
                    pds_data.get('extraction_success', True),
                    json.dumps(pds_data.get('extraction_errors', [])),
                    pds_data.get('processing_notes')
                ))
                result = cursor.fetchone()
                conn.commit()
                return result['id']
    
    def get_pds_candidate(self, candidate_id: int) -> Optional[Dict]:
        """Get a PDS candidate by ID"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT pc.*, j.title as job_title, j.department, j.description as job_description
                    FROM pds_candidates pc
                    LEFT JOIN jobs j ON pc.job_id = j.id
                    WHERE pc.id = %s
                ''', (candidate_id,))
                row = cursor.fetchone()
                if row:
                    candidate = dict(row)
                    # Parse JSON fields safely
                    json_fields = ['personal_info', 'family_background', 'educational_background',
                                 'civil_service_eligibility', 'work_experience', 'voluntary_work',
                                 'learning_development', 'other_information', 'personal_references',
                                 'government_ids', 'scoring_breakdown', 'matched_qualifications',
                                 'areas_for_improvement', 'extraction_errors']
                    
                    for field in json_fields:
                        try:
                            if isinstance(candidate.get(field), str):
                                candidate[field] = json.loads(candidate[field])
                            elif candidate.get(field) is None:
                                if field in ['personal_info', 'family_background', 'educational_background', 
                                           'other_information', 'government_ids', 'scoring_breakdown']:
                                    candidate[field] = {}
                                else:
                                    candidate[field] = []
                        except (json.JSONDecodeError, TypeError):
                            if field in ['personal_info', 'family_background', 'educational_background', 
                                       'other_information', 'government_ids', 'scoring_breakdown']:
                                candidate[field] = {}
                            else:
                                candidate[field] = []
                    
                    return candidate
                return None
    
    def get_all_pds_candidates(self) -> List[Dict]:
        """Get all PDS candidates"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT pc.*, j.title as job_title, j.department
                    FROM pds_candidates pc
                    LEFT JOIN jobs j ON pc.job_id = j.id
                    ORDER BY pc.score DESC, pc.created_at DESC
                ''')
                candidates = []
                for row in cursor.fetchall():
                    candidate = dict(row)
                    # Parse JSON fields safely (lightweight version for list view)
                    try:
                        candidate['personal_info'] = json.loads(candidate['personal_info']) if candidate['personal_info'] else {}
                        candidate['scoring_breakdown'] = json.loads(candidate['scoring_breakdown']) if candidate['scoring_breakdown'] else {}
                    except (json.JSONDecodeError, TypeError):
                        candidate['personal_info'] = {}
                        candidate['scoring_breakdown'] = {}
                    
                    candidates.append(candidate)
                return candidates
    
    def get_pds_candidates_by_job(self, job_id: int) -> List[Dict]:
        """Get PDS candidates for a specific job"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    SELECT pc.*, j.title as job_title, j.department
                    FROM pds_candidates pc
                    LEFT JOIN jobs j ON pc.job_id = j.id
                    WHERE pc.job_id = %s
                    ORDER BY pc.score DESC, pc.created_at DESC
                ''', (job_id,))
                candidates = []
                for row in cursor.fetchall():
                    candidate = dict(row)
                    try:
                        candidate['personal_info'] = json.loads(candidate['personal_info']) if candidate['personal_info'] else {}
                        candidate['scoring_breakdown'] = json.loads(candidate['scoring_breakdown']) if candidate['scoring_breakdown'] else {}
                    except (json.JSONDecodeError, TypeError):
                        candidate['personal_info'] = {}
                        candidate['scoring_breakdown'] = {}
                    
                    candidates.append(candidate)
                return candidates
    
    def update_pds_candidate(self, candidate_id: int, pds_data: Dict) -> bool:
        """Update a PDS candidate"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                fields = []
                values = []
                
                simple_fields = ['name', 'email', 'phone', 'status', 'score', 'job_id', 
                               'highest_education', 'years_of_experience', 'government_service_years',
                               'civil_service_eligible', 'processing_notes']
                
                json_fields = ['personal_info', 'family_background', 'educational_background',
                             'civil_service_eligibility', 'work_experience', 'voluntary_work',
                             'learning_development', 'other_information', 'personal_references',
                             'government_ids', 'scoring_breakdown', 'matched_qualifications',
                             'areas_for_improvement', 'extraction_errors']
                
                for key, value in pds_data.items():
                    if key in simple_fields:
                        fields.append(f"{key} = %s")
                        values.append(value)
                    elif key in json_fields:
                        fields.append(f"{key} = %s")
                        values.append(json.dumps(value))
                
                if not fields:
                    return False
                
                fields.append("updated_at = CURRENT_TIMESTAMP")
                values.append(candidate_id)
                
                query = f"UPDATE pds_candidates SET {', '.join(fields)} WHERE id = %s"
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
    
    def delete_pds_candidate(self, candidate_id: int) -> bool:
        """Delete a PDS candidate"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM pds_candidates WHERE id = %s", (candidate_id,))
                conn.commit()
                return cursor.rowcount > 0
    
    def get_pds_analytics_summary(self) -> Dict:
        """Get PDS analytics summary data"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Basic counts
                cursor.execute("SELECT COUNT(*) as total_pds FROM pds_candidates")
                total_pds = cursor.fetchone()['total_pds']
                
                cursor.execute("SELECT COUNT(*) as processed FROM pds_candidates WHERE status != 'new'")
                processed_pds = cursor.fetchone()['processed']
                
                cursor.execute("SELECT COUNT(*) as shortlisted FROM pds_candidates WHERE status = 'shortlisted'")
                shortlisted = cursor.fetchone()['shortlisted']
                
                cursor.execute("SELECT COUNT(*) as rejected FROM pds_candidates WHERE status = 'rejected'")
                rejected = cursor.fetchone()['rejected']
                
                cursor.execute("SELECT COUNT(*) as civil_service_eligible FROM pds_candidates WHERE civil_service_eligible = TRUE")
                civil_service_eligible = cursor.fetchone()['civil_service_eligible']
                
                # Average score
                cursor.execute("SELECT AVG(score) as avg_score FROM pds_candidates")
                avg_score_result = cursor.fetchone()['avg_score']
                avg_score = float(avg_score_result) if avg_score_result else 0.0
                
                # Education level distribution
                cursor.execute("SELECT highest_education, COUNT(*) as count FROM pds_candidates WHERE highest_education IS NOT NULL GROUP BY highest_education")
                education_stats = {row['highest_education']: row['count'] for row in cursor.fetchall()}
                
                return {
                    'total_pds': total_pds,
                    'processed_pds': processed_pds,
                    'shortlisted': shortlisted,
                    'rejected': rejected,
                    'civil_service_eligible': civil_service_eligible,
                    'avg_score': round(avg_score, 2),
                    'education_stats': education_stats
                }
    
    # ============================================================================
    # UNIVERSITY ASSESSMENT SYSTEM METHODS
    # ============================================================================
    
    def get_position_types(self) -> List[Dict]:
        """Get all position types"""
        if self.use_sqlite and self.sqlite_assessment:
            return self.sqlite_assessment.get_position_types()
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, name, description, is_active, created_at, updated_at
                    FROM position_types
                    WHERE is_active = TRUE
                    ORDER BY name
                """)
                return [dict(row) for row in cursor.fetchall()]
    
    def get_position_type(self, position_type_id: int) -> Optional[Dict]:
        """Get a specific position type"""
        if self.use_sqlite and self.sqlite_assessment:
            return self.sqlite_assessment.get_position_type(position_type_id)
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, name, description, is_active, created_at, updated_at
                    FROM position_types
                    WHERE id = %s
                """, (position_type_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
    
    def get_assessment_templates(self, position_type_id: int) -> List[Dict]:
        """Get assessment templates for a position type"""
        if self.use_sqlite and self.sqlite_assessment:
            return self.sqlite_assessment.get_assessment_templates(position_type_id)
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, position_type_id, criteria_category, criteria_name, 
                           max_points, weight_percentage, scoring_rules, is_automated,
                           display_order, description, created_at, updated_at
                    FROM assessment_templates
                    WHERE position_type_id = %s
                    ORDER BY display_order, criteria_category, criteria_name
                """, (position_type_id,))
                templates = []
                for row in cursor.fetchall():
                    template = dict(row)
                    # Parse JSON scoring_rules - handle both string and dict
                    if template['scoring_rules']:
                        if isinstance(template['scoring_rules'], str):
                            try:
                                template['scoring_rules'] = json.loads(template['scoring_rules'])
                            except json.JSONDecodeError:
                                template['scoring_rules'] = {}
                        elif not isinstance(template['scoring_rules'], dict):
                            template['scoring_rules'] = {}
                    templates.append(template)
                return templates
    
    def get_assessment_templates_by_category(self, position_type_id: int) -> Dict[str, List[Dict]]:
        """Get assessment templates grouped by category"""
        if self.use_sqlite and self.sqlite_assessment:
            return self.sqlite_assessment.get_assessment_templates_by_category(position_type_id)
        
        templates = self.get_assessment_templates(position_type_id)
        categorized = {}
        for template in templates:
            category = template['criteria_category']
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(template)
        return categorized
    
    def create_position_requirement(self, job_id: int, position_type_id: int, 
                                   minimum_education: str = None, required_experience: int = 0,
                                   required_certifications: List[str] = None, 
                                   preferred_qualifications: str = None,
                                   subject_area: str = None) -> int:
        """Create position requirements for a job"""
        if self.use_sqlite and self.sqlite_assessment:
            return self.sqlite_assessment.create_position_requirement(
                job_id, position_type_id, minimum_education, required_experience,
                required_certifications, preferred_qualifications, subject_area
            )
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO position_requirements 
                    (job_id, position_type_id, minimum_education, required_experience, 
                     required_certifications, preferred_qualifications, subject_area)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (job_id, position_type_id, minimum_education, required_experience,
                      json.dumps(required_certifications or []), preferred_qualifications, subject_area))
                
                result = cursor.fetchone()
                conn.commit()
                return result['id'] if result else None
    
    def get_position_requirements(self, job_id: int) -> Optional[Dict]:
        """Get position requirements for a job"""
        if self.use_sqlite and self.sqlite_assessment:
            return self.sqlite_assessment.get_position_requirements(job_id)
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT pr.*, pt.name as position_type_name
                    FROM position_requirements pr
                    JOIN position_types pt ON pr.position_type_id = pt.id
                    WHERE pr.job_id = %s
                """, (job_id,))
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    # Parse JSON required_certifications - handle both string and dict
                    if result['required_certifications']:
                        if isinstance(result['required_certifications'], str):
                            try:
                                result['required_certifications'] = json.loads(result['required_certifications'])
                            except json.JSONDecodeError:
                                result['required_certifications'] = []
                        elif not isinstance(result['required_certifications'], list):
                            result['required_certifications'] = []
                    return result
                return None
    
    def create_candidate_assessment(self, candidate_id: int, job_id: int, 
                                   position_type_id: int, assessed_by: int = None) -> int:
        """Create a new candidate assessment record"""
        if self.use_sqlite and self.sqlite_assessment:
            return self.sqlite_assessment.create_candidate_assessment(
                candidate_id, job_id, position_type_id, assessed_by
            )
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO candidate_assessments 
                    (candidate_id, job_id, position_type_id, assessed_by)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (candidate_id, job_id) 
                    DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (candidate_id, job_id, position_type_id, assessed_by))
                
                result = cursor.fetchone()
                conn.commit()
                return result['id'] if result else None
    
    def get_candidate_assessment(self, candidate_id: int, job_id: int) -> Optional[Dict]:
        """Get candidate assessment by candidate and job"""
        if self.use_sqlite and self.sqlite_assessment:
            return self.sqlite_assessment.get_candidate_assessment(candidate_id, job_id)
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT ca.*, c.name as candidate_name, j.title as job_title, 
                           pt.name as position_type_name
                    FROM candidate_assessments ca
                    JOIN candidates c ON ca.candidate_id = c.id
                    JOIN jobs j ON ca.job_id = j.id
                    JOIN position_types pt ON ca.position_type_id = pt.id
                    WHERE ca.candidate_id = %s AND ca.job_id = %s
                """, (candidate_id, job_id))
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    # Parse JSON score_breakdown - handle both string and dict
                    if result['score_breakdown']:
                        if isinstance(result['score_breakdown'], str):
                            try:
                                result['score_breakdown'] = json.loads(result['score_breakdown'])
                            except json.JSONDecodeError:
                                result['score_breakdown'] = {}
                        elif not isinstance(result['score_breakdown'], dict):
                            result['score_breakdown'] = {}
                    return result
                return None
    
    def get_candidate_assessment_by_id(self, assessment_id: int) -> Optional[Dict]:
        """Get candidate assessment by assessment ID"""
        if self.use_sqlite and self.sqlite_assessment:
            return self.sqlite_assessment.get_candidate_assessment_by_id(assessment_id)
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT ca.*, c.name as candidate_name, c.email as candidate_email,
                           j.title as job_title, pt.name as position_type_name
                    FROM candidate_assessments ca
                    JOIN candidates c ON ca.candidate_id = c.id
                    JOIN jobs j ON ca.job_id = j.id
                    JOIN position_types pt ON ca.position_type_id = pt.id
                    WHERE ca.id = %s
                """, (assessment_id,))
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    # Parse JSON score_breakdown - handle both string and dict
                    if result['score_breakdown']:
                        if isinstance(result['score_breakdown'], str):
                            try:
                                result['score_breakdown'] = json.loads(result['score_breakdown'])
                            except json.JSONDecodeError:
                                result['score_breakdown'] = {}
                        elif not isinstance(result['score_breakdown'], dict):
                            result['score_breakdown'] = {}
                    return result
                return None
    
    def update_candidate_assessment_scores(self, assessment_id: int, 
                                          education_score: float = None,
                                          experience_score: float = None,
                                          training_score: float = None,
                                          eligibility_score: float = None,
                                          accomplishments_score: float = None,
                                          interview_score: float = None,
                                          aptitude_score: float = None,
                                          score_breakdown: Dict = None,
                                          assessment_notes: str = None) -> bool:
        """Update assessment scores for a candidate"""
        if self.use_sqlite and self.sqlite_assessment:
            return self.sqlite_assessment.update_candidate_assessment_scores(
                assessment_id, education_score, experience_score, training_score,
                eligibility_score, accomplishments_score, interview_score,
                aptitude_score, score_breakdown, assessment_notes
            )
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Build dynamic update query based on provided values
                updates = []
                values = []
                
                if education_score is not None:
                    updates.append("education_score = %s")
                    values.append(education_score)
                
                if experience_score is not None:
                    updates.append("experience_score = %s")
                    values.append(experience_score)
                
                if training_score is not None:
                    updates.append("training_score = %s")
                    values.append(training_score)
                
                if eligibility_score is not None:
                    updates.append("eligibility_score = %s")
                    values.append(eligibility_score)
                
                if accomplishments_score is not None:
                    updates.append("accomplishments_score = %s")
                    values.append(accomplishments_score)
                
                if interview_score is not None:
                    updates.append("interview_score = %s")
                    values.append(interview_score)
                
                if aptitude_score is not None:
                    updates.append("aptitude_score = %s")
                    values.append(aptitude_score)
                
                if score_breakdown is not None:
                    updates.append("score_breakdown = %s")
                    values.append(json.dumps(score_breakdown))
                
                if assessment_notes is not None:
                    updates.append("assessment_notes = %s")
                    values.append(assessment_notes)
                
                # Always update automated_total, manual_total, final_score, and timestamp
                updates.extend([
                    "automated_total = COALESCE(education_score, 0) + COALESCE(experience_score, 0) + COALESCE(training_score, 0) + COALESCE(eligibility_score, 0) + COALESCE(accomplishments_score, 0)",
                    "manual_total = COALESCE(interview_score, 0) + COALESCE(aptitude_score, 0)",
                    "final_score = automated_total + manual_total",
                    "updated_at = CURRENT_TIMESTAMP"
                ])
                
                values.append(assessment_id)
                
                query = f"""
                    UPDATE candidate_assessments 
                    SET {', '.join(updates)}
                    WHERE id = %s
                """
                
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
    
    def update_assessment_status(self, assessment_id: int, status: str, 
                                recommendation: str = None, completed_date: datetime = None) -> bool:
        """Update assessment status and recommendation"""
        if self.use_sqlite and self.sqlite_assessment:
            return self.sqlite_assessment.update_assessment_status(
                assessment_id, status, recommendation, completed_date
            )
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE candidate_assessments 
                    SET assessment_status = %s, recommendation = %s, completed_date = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (status, recommendation, completed_date, assessment_id))
                
                conn.commit()
                return cursor.rowcount > 0
    
    def get_assessments_for_job(self, job_id: int, status: str = None) -> List[Dict]:
        """Get all assessments for a job with optional status filter"""
        if self.use_sqlite and self.sqlite_assessment:
            return self.sqlite_assessment.get_assessments_for_job(job_id, status)
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                base_query = """
                    SELECT ca.*, c.name as candidate_name, c.email as candidate_email,
                           pt.name as position_type_name, u.username as assessor_name
                    FROM candidate_assessments ca
                    JOIN candidates c ON ca.candidate_id = c.id
                    JOIN position_types pt ON ca.position_type_id = pt.id
                    LEFT JOIN users u ON ca.assessed_by = u.id
                    WHERE ca.job_id = %s
                """
                
                params = [job_id]
                if status:
                    base_query += " AND ca.assessment_status = %s"
                    params.append(status)
                
                base_query += " ORDER BY ca.final_score DESC, ca.updated_at DESC"
                
                cursor.execute(base_query, params)
                assessments = []
                for row in cursor.fetchall():
                    assessment = dict(row)
                    # Parse JSON score_breakdown
                    if assessment['score_breakdown']:
                        try:
                            assessment['score_breakdown'] = json.loads(assessment['score_breakdown'])
                        except json.JSONDecodeError:
                            assessment['score_breakdown'] = {}
                    assessments.append(assessment)
                
                return assessments
    
    def create_manual_assessment_score(self, candidate_assessment_id: int, score_type: str,
                                      component_name: str, rating: int, score: float,
                                      max_possible: float, notes: str = None, 
                                      entered_by: int = None) -> int:
        """Create a manual assessment score entry"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO manual_assessment_scores 
                    (candidate_assessment_id, score_type, component_name, rating, 
                     score, max_possible, notes, entered_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (candidate_assessment_id, score_type, component_name, rating,
                      score, max_possible, notes, entered_by))
                
                result = cursor.fetchone()
                conn.commit()
                return result['id'] if result else None
    
    def get_manual_assessment_scores(self, candidate_assessment_id: int) -> List[Dict]:
        """Get all manual assessment scores for a candidate assessment"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT mas.*, u.username as entered_by_name
                    FROM manual_assessment_scores mas
                    LEFT JOIN users u ON mas.entered_by = u.id
                    WHERE mas.candidate_assessment_id = %s
                    ORDER BY mas.score_type, mas.component_name
                """, (candidate_assessment_id,))
                return [dict(row) for row in cursor.fetchall()]
    
    def update_assessment_rankings(self, job_id: int) -> bool:
        """Update rank positions for all assessments in a job based on final scores"""
        if self.use_sqlite and self.sqlite_assessment:
            return self.sqlite_assessment.update_assessment_rankings(job_id)
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                # Update rankings based on final_score descending
                cursor.execute("""
                    UPDATE candidate_assessments 
                    SET rank_position = ranked.rank
                    FROM (
                        SELECT id, ROW_NUMBER() OVER (ORDER BY final_score DESC, updated_at ASC) as rank
                        FROM candidate_assessments 
                        WHERE job_id = %s AND assessment_status != 'incomplete'
                    ) as ranked
                    WHERE candidate_assessments.id = ranked.id
                """, (job_id,))
                
                conn.commit()
                return cursor.rowcount > 0
    
    def save_assessment_comparison(self, job_id: int, candidate_rankings: List[Dict],
                                  assessment_summary: Dict, generated_by: int) -> int:
        """Save assessment comparison results"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO assessment_comparisons 
                    (job_id, candidate_rankings, assessment_summary, generated_by)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (job_id, json.dumps(candidate_rankings), 
                      json.dumps(assessment_summary), generated_by))
                
                result = cursor.fetchone()
                conn.commit()
                return result['id'] if result else None
    
    def get_assessment_comparison(self, job_id: int, latest: bool = True) -> Optional[Dict]:
        """Get assessment comparison results for a job"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                base_query = """
                    SELECT ac.*, u.username as generated_by_name
                    FROM assessment_comparisons ac
                    LEFT JOIN users u ON ac.generated_by = u.id
                    WHERE ac.job_id = %s
                """
                
                if latest:
                    base_query += " ORDER BY ac.comparison_date DESC LIMIT 1"
                
                cursor.execute(base_query, (job_id,))
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    # Parse JSON fields
                    try:
                        result['candidate_rankings'] = json.loads(result['candidate_rankings'])
                        if result['assessment_summary']:
                            result['assessment_summary'] = json.loads(result['assessment_summary'])
                    except json.JSONDecodeError:
                        result['candidate_rankings'] = []
                        result['assessment_summary'] = {}
                    return result
                return None
    
    def get_assessment_analytics(self, job_id: int = None, position_type_id: int = None) -> Dict:
        """Get assessment analytics and statistics"""
        if self.use_sqlite and self.sqlite_assessment:
            return self.sqlite_assessment.get_assessment_analytics(job_id, position_type_id)
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                base_where = []
                params = []
                
                if job_id:
                    base_where.append("ca.job_id = %s")
                    params.append(job_id)
                
                if position_type_id:
                    base_where.append("ca.position_type_id = %s")
                    params.append(position_type_id)
                
                where_clause = " AND ".join(base_where)
                if where_clause:
                    where_clause = "WHERE " + where_clause
                
                # Basic counts
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total_assessments,
                        COUNT(CASE WHEN assessment_status = 'complete' THEN 1 END) as completed,
                        COUNT(CASE WHEN assessment_status = 'pending_interview' THEN 1 END) as pending_interview,
                        COUNT(CASE WHEN assessment_status = 'incomplete' THEN 1 END) as incomplete,
                        AVG(final_score) as avg_final_score,
                        AVG(automated_total) as avg_automated_score,
                        AVG(manual_total) as avg_manual_score
                    FROM candidate_assessments ca
                    {where_clause}
                """, params)
                
                stats = dict(cursor.fetchone())
                
                # Score distribution
                cursor.execute(f"""
                    SELECT 
                        CASE 
                            WHEN final_score >= 90 THEN 'Excellent (90+)'
                            WHEN final_score >= 80 THEN 'Very Good (80-89)'
                            WHEN final_score >= 70 THEN 'Good (70-79)'
                            WHEN final_score >= 60 THEN 'Fair (60-69)'
                            ELSE 'Needs Improvement (<60)'
                        END as score_range,
                        COUNT(*) as count
                    FROM candidate_assessments ca
                    {where_clause} AND assessment_status != 'incomplete'
                    GROUP BY 1
                    ORDER BY MIN(final_score) DESC
                """, params)
                
                score_distribution = {row['score_range']: row['count'] for row in cursor.fetchall()}
                
                # Recommendation distribution
                cursor.execute(f"""
                    SELECT recommendation, COUNT(*) as count
                    FROM candidate_assessments ca
                    {where_clause} AND assessment_status = 'complete'
                    GROUP BY recommendation
                """, params)
                
                recommendation_stats = {row['recommendation']: row['count'] for row in cursor.fetchall()}
                
                return {
                    'total_assessments': stats['total_assessments'],
                    'completed': stats['completed'],
                    'pending_interview': stats['pending_interview'],
                    'incomplete': stats['incomplete'],
                    'avg_final_score': round(float(stats['avg_final_score']) if stats['avg_final_score'] else 0, 2),
                    'avg_automated_score': round(float(stats['avg_automated_score']) if stats['avg_automated_score'] else 0, 2),
                    'avg_manual_score': round(float(stats['avg_manual_score']) if stats['avg_manual_score'] else 0, 2),
                    'score_distribution': score_distribution,
                    'recommendation_stats': recommendation_stats
                }

    # Upload Session Management Methods
    def create_upload_session(self, session_id: str, user_id: int, job_id: int) -> bool:
        """Create a new upload session"""
        try:
            if self.use_sqlite:
                import sqlite3
                conn = sqlite3.connect('resume_screening.db')
                cursor = conn.cursor()
                
                # Create table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS upload_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        user_id INTEGER,
                        job_id INTEGER NOT NULL,
                        status TEXT DEFAULT 'pending',
                        file_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP,
                        session_data TEXT,
                        error_log TEXT
                    )
                ''')
                
                cursor.execute('''
                    INSERT INTO upload_sessions (session_id, user_id, job_id, status)
                    VALUES (?, ?, ?, 'pending')
                ''', (session_id, user_id, job_id))
                
                conn.commit()
                conn.close()
                return True
            else:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO upload_sessions (session_id, user_id, job_id, status)
                        VALUES (%s, %s, %s, 'pending')
                    ''', (session_id, user_id, job_id))
                    conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error creating upload session: {e}")
            return False

    def get_upload_session(self, session_id: str) -> Optional[Dict]:
        """Get upload session by session_id"""
        try:
            if self.use_sqlite:
                import sqlite3
                conn = sqlite3.connect('resume_screening.db')
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM upload_sessions WHERE session_id = ?
                ''', (session_id,))
                
                row = cursor.fetchone()
                conn.close()
                return dict(row) if row else None
            else:
                with self.get_connection() as conn:
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                    cursor.execute('''
                        SELECT * FROM upload_sessions WHERE session_id = %s
                    ''', (session_id,))
                    row = cursor.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting upload session: {e}")
            return None

    def update_upload_session(self, session_id: str, **kwargs) -> bool:
        """Update upload session with provided fields"""
        try:
            if not kwargs:
                return True
            
            set_clauses = []
            values = []
            
            for key, value in kwargs.items():
                if key in ['status', 'file_count', 'completed_at', 'session_data', 'error_log']:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
            
            if not set_clauses:
                return True
            
            values.append(session_id)
            
            if self.use_sqlite:
                import sqlite3
                conn = sqlite3.connect('resume_screening.db')
                cursor = conn.cursor()
                
                query = f"UPDATE upload_sessions SET {', '.join(set_clauses)} WHERE session_id = ?"
                cursor.execute(query, values)
                
                conn.commit()
                conn.close()
                return True
            else:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Convert ? to %s for PostgreSQL
                    query = f"UPDATE upload_sessions SET {', '.join(set_clauses).replace('?', '%s')} WHERE session_id = %s"
                    cursor.execute(query, values)
                    conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating upload session: {e}")
            return False

    def create_upload_file_record(self, session_id: str, file_id: str, file_data: Dict) -> bool:
        """Create a record for an uploaded file"""
        try:
            if self.use_sqlite:
                import sqlite3
                conn = sqlite3.connect('resume_screening.db')
                cursor = conn.cursor()
                
                # Create table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS upload_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        file_id TEXT UNIQUE NOT NULL,
                        original_name TEXT NOT NULL,
                        temp_path TEXT NOT NULL,
                        file_size INTEGER,
                        file_type TEXT,
                        status TEXT DEFAULT 'uploaded',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processed_at TIMESTAMP,
                        candidate_id INTEGER,
                        error_message TEXT
                    )
                ''')
                
                cursor.execute('''
                    INSERT INTO upload_files 
                    (session_id, file_id, original_name, temp_path, file_size, file_type, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'uploaded')
                ''', (
                    session_id,
                    file_id,
                    file_data.get('original_name'),
                    file_data.get('temp_path'),
                    file_data.get('size'),
                    file_data.get('type'),
                ))
                
                conn.commit()
                conn.close()
                return True
            else:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO upload_files 
                        (session_id, filename, file_size, file_type, status)
                        VALUES (%s, %s, %s, %s, 'uploaded')
                    ''', (
                        session_id,
                        file_data.get('original_name'),
                        file_data.get('size'),
                        file_data.get('type'),
                    ))
                    conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error creating upload file record: {e}")
            return False

    def get_upload_files(self, session_id: str) -> List[Dict]:
        """Get all files for an upload session with metadata"""
        try:
            # Get files from database
            if self.use_sqlite:
                import sqlite3
                conn = sqlite3.connect('resume_screening.db')
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM upload_files WHERE session_id = ? ORDER BY created_at
                ''', (session_id,))
                
                rows = cursor.fetchall()
                files = [dict(row) for row in rows]
                conn.close()
            else:
                with self.get_connection() as conn:
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                    cursor.execute('''
                        SELECT * FROM upload_files WHERE session_id = %s ORDER BY id
                    ''', (session_id,))
                    files = [dict(row) for row in cursor.fetchall()]
            
            # Get session metadata to enrich file data
            session = self.get_upload_session(session_id)
            if session and session.get('metadata'):
                try:
                    session_data = json.loads(session['metadata'])
                    file_metadata = session_data.get('file_metadata', {})
                    
                    # Enrich files with metadata
                    for file_record in files:
                        # Map filename back to file_id (stored in metadata)
                        for file_id, metadata in file_metadata.items():
                            if metadata.get('original_name') == file_record.get('filename'):
                                file_record.update({
                                    'file_id': file_id,
                                    'temp_path': metadata.get('temp_path'),
                                    'original_name': metadata.get('original_name')
                                })
                                break
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse session metadata for {session_id}")
            
            return files
        except Exception as e:
            logger.error(f"Error getting upload files: {e}")
            return []

    def update_upload_file_status(self, file_id: str, status: str, candidate_id: int = None, error_message: str = None) -> bool:
        """Update upload file processing status"""
        try:
            if self.use_sqlite:
                import sqlite3
                conn = sqlite3.connect('resume_screening.db')
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE upload_files 
                    SET status = ?, processed_at = CURRENT_TIMESTAMP, candidate_id = ?, error_message = ?
                    WHERE file_id = ?
                ''', (status, candidate_id, error_message, file_id))
                
                conn.commit()
                conn.close()
                return True
            else:
                # For PostgreSQL, find the file by filename (since we don't have file_id column)
                # This is a workaround - ideally we'd modify the schema
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE upload_files 
                        SET status = %s, processed_at = CURRENT_TIMESTAMP, candidate_id = %s, error_message = %s
                        WHERE id = (SELECT id FROM upload_files WHERE filename LIKE %s LIMIT 1)
                    ''', (status, candidate_id, error_message, f'%{file_id}%'))
                    conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating upload file status: {e}")
            return False