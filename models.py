from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class JobCategory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    requirements = db.Column(db.Text, nullable=False)
    experience_level = db.Column(db.String(20), nullable=False, default='mid')
    category_id = db.Column(db.Integer, db.ForeignKey('job_category.id'), nullable=False)
    category = db.relationship('JobCategory', backref=db.backref('jobs', lazy=True))
    status = db.Column(db.String(20), default='active')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'department': self.department,
            'description': self.description,
            'requirements': self.requirements,
            'experience_level': self.experience_level,
            'category': self.category.name if self.category else None,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    linkedin = db.Column(db.String(200))
    github = db.Column(db.String(200))
    resume_text = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50))
    skills = db.Column(db.Text)
    education = db.Column(db.Text)  # JSON string
    experience = db.Column(db.Text)  # JSON string
    status = db.Column(db.String(20), default='new')  # new, reviewed, shortlisted, rejected, hired
    score = db.Column(db.Float, default=0.0)
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=True)
    job = db.relationship('Job', backref=db.backref('candidates', lazy=True))
    notes = db.Column(db.Text)  # HR notes
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        import json
        
        # Parse JSON fields safely
        try:
            education = json.loads(self.education) if self.education else []
        except:
            education = []
            
        try:
            experience = json.loads(self.experience) if self.experience else []
        except:
            experience = []
        
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'linkedin': self.linkedin,
            'github': self.github,
            'category': self.category,
            'skills': self.skills.split(',') if self.skills else [],
            'education': education,
            'experience': experience,
            'status': self.status,
            'score': self.score,
            'job_id': self.job_id,
            'notes': self.notes,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Analytics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    total_resumes = db.Column(db.Integer, default=0)
    processed_resumes = db.Column(db.Integer, default=0)
    shortlisted = db.Column(db.Integer, default=0)
    rejected = db.Column(db.Integer, default=0)
    avg_processing_time = db.Column(db.Float, default=0.0)
    job_category_stats = db.Column(db.Text)  # JSON string

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