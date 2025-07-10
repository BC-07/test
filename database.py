import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.getenv('DATABASE_URL')
        if not self.db_url:
            # Fallback to individual components
            db_user = os.getenv('DB_USER')
            db_password = os.getenv('DB_PASSWORD')
            db_host = os.getenv('DB_HOST', 'localhost')
            db_port = os.getenv('DB_PORT', '5432')
            db_name = os.getenv('DB_NAME')
            
            if not all([db_user, db_password, db_name]):
                raise ValueError("Database configuration missing. Please set DATABASE_URL or individual DB_* environment variables.")
            
            self.db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
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
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO candidates 
                    (name, email, phone, linkedin, github, resume_text, category, skills, 
                     education, experience, status, score, job_id, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    candidate_data.get('notes')
                ))
                result = cursor.fetchone()
                conn.commit()
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
                    try:
                        if isinstance(candidate['education'], str):
                            candidate['education'] = json.loads(candidate['education'])
                        elif candidate['education'] is None:
                            candidate['education'] = []
                    except (json.JSONDecodeError, TypeError):
                        candidate['education'] = []
                    
                    try:
                        if isinstance(candidate['experience'], str):
                            candidate['experience'] = json.loads(candidate['experience'])
                        elif candidate['experience'] is None:
                            candidate['experience'] = []
                    except (json.JSONDecodeError, TypeError):
                        candidate['experience'] = []
                    
                    candidate['skills'] = candidate['skills'].split(',') if candidate['skills'] else []
                    return candidate
                return None
    
    def get_all_candidates(self) -> List[Dict]:
        """Get all candidates"""
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
                    try:
                        if isinstance(candidate['education'], str):
                            candidate['education'] = json.loads(candidate['education'])
                        elif candidate['education'] is None:
                            candidate['education'] = []
                    except (json.JSONDecodeError, TypeError):
                        candidate['education'] = []
                    
                    try:
                        if isinstance(candidate['experience'], str):
                            candidate['experience'] = json.loads(candidate['experience'])
                        elif candidate['experience'] is None:
                            candidate['experience'] = []
                    except (json.JSONDecodeError, TypeError):
                        candidate['experience'] = []
                    
                    candidate['skills'] = candidate['skills'].split(',') if candidate['skills'] else []
                    candidates.append(candidate)
                return candidates
    
    def update_candidate(self, candidate_id: int, candidate_data: Dict) -> bool:
        """Update a candidate"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                fields = []
                values = []
                
                for key, value in candidate_data.items():
                    if key in ['name', 'email', 'phone', 'linkedin', 'github', 'status', 'score', 'job_id', 'notes']:
                        fields.append(f"{key} = %s")
                        values.append(value)
                    elif key in ['education', 'experience']:
                        fields.append(f"{key} = %s")
                        values.append(json.dumps(value))
                
                if not fields:
                    return False
                
                fields.append("updated_at = CURRENT_TIMESTAMP")
                values.append(candidate_id)
                
                query = f"UPDATE candidates SET {', '.join(fields)} WHERE id = %s"
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
    
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
                    'shortlisted': shortlisted,
                    'rejected': rejected,
                    'avg_score': round(avg_score, 2),
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