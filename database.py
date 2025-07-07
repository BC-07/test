import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "resume_screening.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection with foreign key support"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn
    
    def init_database(self):
        """Initialize database with all required tables"""
        with self.get_connection() as conn:
            # Jobs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    department TEXT NOT NULL,
                    category TEXT NOT NULL,
                    experience_level TEXT NOT NULL,
                    description TEXT NOT NULL,
                    requirements TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Job categories table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS job_categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Candidates table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT,
                    phone TEXT,
                    education TEXT,  -- JSON string
                    all_skills TEXT,  -- JSON string
                    filename TEXT NOT NULL,
                    job_id INTEGER,
                    job_title TEXT,
                    category TEXT,
                    predicted_category TEXT,  -- JSON string
                    score INTEGER,
                    matched_skills TEXT,  -- JSON string
                    missing_skills TEXT,  -- JSON string
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES jobs (id) ON DELETE CASCADE
                )
            ''')
            
            # Settings table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert default job if none exists
            cursor = conn.execute("SELECT COUNT(*) FROM jobs")
            if cursor.fetchone()[0] == 0:
                conn.execute('''
                    INSERT INTO jobs (title, department, category, experience_level, description, requirements)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    'Software Developer',
                    'Engineering',
                    'Software Development',
                    'mid',
                    'We are looking for a skilled software developer to join our team. You can modify this job or create new ones in the Job Requirements section.',
                    'Python, JavaScript, React, Node.js, SQL, Git'
                ))
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    # Job operations
    def create_job(self, job_data: Dict) -> int:
        """Create a new job and return its ID"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO jobs (title, department, category, experience_level, description, requirements)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                job_data['title'],
                job_data['department'],
                job_data['category'],
                job_data['experience_level'],
                job_data['description'],
                job_data['requirements']
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_job(self, job_id: int) -> Optional[Dict]:
        """Get a job by ID"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_all_jobs(self) -> List[Dict]:
        """Get all jobs"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    def update_job(self, job_id: int, job_data: Dict) -> bool:
        """Update a job"""
        with self.get_connection() as conn:
            # Build dynamic update query
            fields = []
            values = []
            for key, value in job_data.items():
                if key in ['title', 'department', 'category', 'experience_level', 'description', 'requirements', 'status']:
                    fields.append(f"{key} = ?")
                    values.append(value)
            
            if not fields:
                return False
            
            fields.append("updated_at = CURRENT_TIMESTAMP")
            values.append(job_id)
            
            query = f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?"
            cursor = conn.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_job(self, job_id: int) -> bool:
        """Delete a job"""
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    # Job category operations
    def create_job_category(self, name: str, description: str = "") -> int:
        """Create a new job category"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO job_categories (name, description)
                VALUES (?, ?)
            ''', (name, description))
            conn.commit()
            return cursor.lastrowid
    
    def get_all_job_categories(self) -> List[Dict]:
        """Get all job categories"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM job_categories ORDER BY name")
            return [dict(row) for row in cursor.fetchall()]
    
    def update_job_category(self, category_id: int, name: str = None, description: str = None) -> bool:
        """Update a job category"""
        with self.get_connection() as conn:
            fields = []
            values = []
            
            if name is not None:
                fields.append("name = ?")
                values.append(name)
            if description is not None:
                fields.append("description = ?")
                values.append(description)
            
            if not fields:
                return False
            
            values.append(category_id)
            query = f"UPDATE job_categories SET {', '.join(fields)} WHERE id = ?"
            cursor = conn.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_job_category(self, category_id: int) -> bool:
        """Delete a job category"""
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM job_categories WHERE id = ?", (category_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def check_category_in_use(self, category_name: str) -> int:
        """Check how many jobs use this category"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM jobs WHERE category = ?", (category_name,))
            return cursor.fetchone()[0]
    
    # Candidate operations
    def create_candidate(self, candidate_data: Dict) -> int:
        """Create a new candidate"""
        with self.get_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO candidates 
                (name, email, phone, education, all_skills, filename, job_id, job_title, 
                 category, predicted_category, score, matched_skills, missing_skills, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                candidate_data['name'],
                candidate_data['email'],
                candidate_data['phone'],
                json.dumps(candidate_data['education']),
                json.dumps(candidate_data['all_skills']),
                candidate_data['filename'],
                candidate_data['job_id'],
                candidate_data['job_title'],
                candidate_data['category'],
                json.dumps(candidate_data['predicted_category']),
                candidate_data['score'],
                json.dumps(candidate_data['matched_skills']),
                json.dumps(candidate_data['missing_skills']),
                candidate_data['status']
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_candidate(self, candidate_id: int) -> Optional[Dict]:
        """Get a candidate by ID"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,))
            row = cursor.fetchone()
            if row:
                candidate = dict(row)
                # Parse JSON fields
                candidate['education'] = json.loads(candidate['education']) if candidate['education'] else []
                candidate['all_skills'] = json.loads(candidate['all_skills']) if candidate['all_skills'] else []
                candidate['predicted_category'] = json.loads(candidate['predicted_category']) if candidate['predicted_category'] else {}
                candidate['matched_skills'] = json.loads(candidate['matched_skills']) if candidate['matched_skills'] else []
                candidate['missing_skills'] = json.loads(candidate['missing_skills']) if candidate['missing_skills'] else []
                return candidate
            return None
    
    def get_all_candidates(self) -> List[Dict]:
        """Get all candidates"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM candidates ORDER BY score DESC")
            candidates = []
            for row in cursor.fetchall():
                candidate = dict(row)
                # Parse JSON fields
                candidate['education'] = json.loads(candidate['education']) if candidate['education'] else []
                candidate['all_skills'] = json.loads(candidate['all_skills']) if candidate['all_skills'] else []
                candidate['predicted_category'] = json.loads(candidate['predicted_category']) if candidate['predicted_category'] else {}
                candidate['matched_skills'] = json.loads(candidate['matched_skills']) if candidate['matched_skills'] else []
                candidate['missing_skills'] = json.loads(candidate['missing_skills']) if candidate['missing_skills'] else []
                candidates.append(candidate)
            return candidates
    
    def update_candidate(self, candidate_id: int, candidate_data: Dict) -> bool:
        """Update a candidate"""
        with self.get_connection() as conn:
            fields = []
            values = []
            
            for key, value in candidate_data.items():
                if key == 'status':
                    fields.append("status = ?")
                    values.append(value)
            
            if not fields:
                return False
            
            fields.append("updated_at = CURRENT_TIMESTAMP")
            values.append(candidate_id)
            
            query = f"UPDATE candidates SET {', '.join(fields)} WHERE id = ?"
            cursor = conn.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_candidate(self, candidate_id: int) -> bool:
        """Delete a candidate"""
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    # Analytics operations
    def get_analytics_summary(self) -> Dict:
        """Get analytics summary data"""
        with self.get_connection() as conn:
            # Basic counts
            cursor = conn.execute("SELECT COUNT(*) FROM candidates")
            total_resumes = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM candidates WHERE status != 'pending'")
            processed_resumes = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM candidates WHERE status = 'shortlisted'")
            shortlisted = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM candidates WHERE status = 'rejected'")
            rejected = cursor.fetchone()[0]
            
            # Average score
            cursor = conn.execute("SELECT AVG(score) FROM candidates")
            avg_score_result = cursor.fetchone()[0]
            avg_score = avg_score_result if avg_score_result else 0
            
            # Job category stats
            cursor = conn.execute("SELECT category, COUNT(*) FROM candidates GROUP BY category")
            job_category_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'total_resumes': total_resumes,
                'processed_resumes': processed_resumes,
                'shortlisted': shortlisted,
                'rejected': rejected,
                'avg_score': round(avg_score, 2),
                'job_category_stats': job_category_stats
            }
    
    # Settings operations
    def get_setting(self, key: str) -> Optional[str]:
        """Get a setting value"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None
    
    def set_setting(self, key: str, value: str):
        """Set a setting value"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, value))
            conn.commit()
    
    def get_all_settings(self) -> Dict:
        """Get all settings"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT key, value FROM settings")
            return {row[0]: row[1] for row in cursor.fetchall()}
    
    def update_settings(self, settings: Dict):
        """Update multiple settings"""
        with self.get_connection() as conn:
            for key, value in settings.items():
                conn.execute('''
                    INSERT OR REPLACE INTO settings (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (key, str(value)))
            conn.commit()
