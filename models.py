from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSONB
import json

db = SQLAlchemy()

class JobCategory(db.Model):
    __tablename__ = 'job_categories'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    jobs = db.relationship('Job', backref='category', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Job(db.Model):
    __tablename__ = 'jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    requirements = db.Column(db.Text, nullable=False)
    experience_level = db.Column(db.String(20), nullable=False, default='mid')
    category_id = db.Column(db.Integer, db.ForeignKey('job_categories.id'), nullable=False)
    status = db.Column(db.String(20), default='active')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    candidates = db.relationship('Candidate', backref='job', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'department': self.department,
            'description': self.description,
            'requirements': self.requirements,
            'experience_level': self.experience_level,
            'category_id': self.category_id,
            'category_name': self.category.name if self.category else None,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Candidate(db.Model):
    __tablename__ = 'candidates'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    linkedin = db.Column(db.String(200))
    github = db.Column(db.String(200))
    resume_text = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50))
    skills = db.Column(db.Text)
    education = db.Column(JSONB)
    experience = db.Column(JSONB)
    status = db.Column(db.String(20), default='new')
    score = db.Column(db.Float, default=0.0)
    job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'), nullable=True)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'linkedin': self.linkedin,
            'github': self.github,
            'category': self.category,
            'skills': self.skills.split(',') if self.skills else [],
            'education': self.education or [],
            'experience': self.experience or [],
            'status': self.status,
            'score': self.score,
            'job_id': self.job_id,
            'job_title': self.job.title if self.job else None,
            'notes': self.notes,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Analytics(db.Model):
    __tablename__ = 'analytics'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    total_resumes = db.Column(db.Integer, default=0)
    processed_resumes = db.Column(db.Integer, default=0)
    shortlisted = db.Column(db.Integer, default=0)
    rejected = db.Column(db.Integer, default=0)
    avg_processing_time = db.Column(db.Float, default=0.0)
    job_category_stats = db.Column(JSONB)

    def to_dict(self):
        return {
            'date': self.date.isoformat(),
            'total_resumes': self.total_resumes,
            'processed_resumes': self.processed_resumes,
            'shortlisted': self.shortlisted,
            'rejected': self.rejected,
            'avg_processing_time': self.avg_processing_time,
            'job_category_stats': self.job_category_stats
        }

class Settings(db.Model):
    __tablename__ = 'settings'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.Text, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'key': self.key,
            'value': self.value,
            'updated_at': self.updated_at.isoformat()
        }