import re
import os
import json
import spacy
import docx2txt
import PyPDF2
import nltk
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')
try:
    nltk.data.find('words')
except LookupError:
    nltk.download('words')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class ResumeProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common skills dictionary
        self.skills_dict = {
            'programming_languages': [
                'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php',
                'swift', 'kotlin', 'go', 'rust', 'typescript'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'django',
                'flask', 'spring', 'asp.net', 'express'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis',
                'elasticsearch', 'cassandra'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
                'jenkins', 'circleci'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'neural networks', 'nlp',
                'computer vision', 'pandas', 'numpy', 'scikit-learn', 'tensorflow',
                'pytorch'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'project management', 'agile', 'scrum'
            ]
        }
        
        # Load skills database
        self.skills_db = self._load_skills_database()
        self.education_keywords = self._load_education_keywords()
        
    def _load_skills_database(self) -> List[str]:
        """Load and return list of technical skills."""
        skills = {
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
            'go', 'rust', 'scala', 'perl', 'r', 'matlab', 'sql', 'bash', 'shell',
            
            # Web Development
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
            'spring', 'asp.net', 'jquery', 'bootstrap', 'sass', 'less', 'webpack', 'gatsby',
            'next.js', 'graphql', 'rest api', 'websocket',
            
            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'oracle',
            'sql server', 'sqlite', 'dynamodb', 'firebase',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'terraform',
            'ansible', 'puppet', 'chef', 'prometheus', 'grafana', 'elk stack',
            
            # AI/ML
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn',
            'pandas', 'numpy', 'scipy', 'opencv', 'nlp', 'computer vision', 'neural networks',
            
            # Mobile Development
            'android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic', 'swift', 'objective-c',
            
            # Tools & Others
            'git', 'svn', 'jira', 'confluence', 'slack', 'trello', 'agile', 'scrum', 'kanban',
            'tdd', 'ci/cd', 'unit testing', 'selenium', 'postman', 'swagger',
            
            # Soft Skills
            'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
            'project management', 'time management', 'critical thinking', 'creativity',
            'adaptability', 'organization', 'presentation', 'negotiation'
        }
        return sorted(list(skills))
        
    def _load_education_keywords(self) -> List[str]:
        """Load and return list of education-related keywords."""
        return [
            # Degrees
            'bachelor', 'master', 'phd', 'doctorate', 'bs', 'ba', 'btech', 'mtech', 'msc',
            'bsc', 'mba', 'associate degree',
            
            # Fields
            'computer science', 'information technology', 'engineering', 'business',
            'mathematics', 'physics', 'data science', 'artificial intelligence',
            'machine learning', 'software engineering', 'electrical engineering',
            
            # Institutions
            'university', 'college', 'institute', 'school',
            
            # Academic Terms
            'major', 'minor', 'concentration', 'specialization', 'thesis',
            'dissertation', 'research', 'academic', 'gpa', 'honors', 'dean\'s list'
        ]
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses for cleaning (we extract them separately)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b(?:\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        return text.strip()
        
    def extract_text_from_file(self, file) -> str:
        """Extract text from various file formats."""
        if hasattr(file, 'filename'):
            # File object
            filename = file.filename.lower()
        else:
            # File path string
            filename = str(file).lower()
        
        try:
            if filename.endswith('.pdf'):
                return self._extract_from_pdf(file)
            elif filename.endswith('.docx'):
                return self._extract_from_docx(file)
            elif filename.endswith('.txt'):
                if hasattr(file, 'read'):
                    return file.read().decode('utf-8')
                else:
                    with open(file, 'r', encoding='utf-8') as f:
                        return f.read()
            else:
                raise ValueError(f"Unsupported file format: {filename}")
        except Exception as e:
            self.logger.error(f"Error extracting text from {filename}: {str(e)}")
            raise
            
    def _extract_from_pdf(self, file) -> str:
        """Extract text from PDF file."""
        try:
            if hasattr(file, 'read'):
                # File object
                pdf_reader = PyPDF2.PdfReader(file)
            else:
                # File path
                with open(file, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
            
    def _extract_from_docx(self, file) -> str:
        """Extract text from DOCX file."""
        try:
            if hasattr(file, 'read'):
                # File object - save temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                    file.seek(0)  # Ensure we're at the beginning
                    tmp.write(file.read())
                    tmp.flush()
                    text = docx2txt.process(tmp.name)
                    os.unlink(tmp.name)  # Clean up
                return text
            else:
                # File path
                return docx2txt.process(file)
        except Exception as e:
            self.logger.error(f"Error extracting text from DOCX: {str(e)}")
            raise
            
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from text."""
        contact_info = {
            'name': '',
            'email': '',
            'phone': '',
            'linkedin': '',
            'github': ''
        }
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        # Extract phone number
        phone_pattern = r'\b(?:\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact_info['phone'] = phone_match.group()
            
        # Extract LinkedIn
        linkedin_pattern = r'linkedin\.com/in/[\w-]+|linkedin\.com/pub/[\w-]+/[\w]+/[\w]+/[\w]+'
        linkedin_match = re.search(linkedin_pattern, text.lower())
        if linkedin_match:
            contact_info['linkedin'] = linkedin_match.group()
            
        # Extract GitHub
        github_pattern = r'github\.com/[\w-]+'
        github_match = re.search(github_pattern, text.lower())
        if github_match:
            contact_info['github'] = github_match.group()
        
        # Extract name using NER
        doc = nlp(text[:1000])  # Process first 1000 chars for efficiency
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                contact_info['name'] = ent.text
                break
        
        return contact_info
        
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text."""
        text_lower = text.lower()
        found_skills = set()
        
        # Extract skills from all categories
        all_skills = []
        for category, skills in self.skills_dict.items():
            all_skills.extend(skills)
        
        # Add skills from skills_db
        all_skills.extend(self.skills_db)
        
        # Remove duplicates and search for skills
        unique_skills = list(set(all_skills))
        
        for skill in unique_skills:
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                found_skills.add(skill)
        
        return sorted(list(found_skills))
        
    def extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information from text."""
        education = []
        
        # Common education keywords
        edu_keywords = r'\b(bachelor|master|phd|doctorate|bsc|msc|be|btech|mtech)\b'
        year_pattern = r'\b(19|20)\d{2}\b'
        
        # Find education sections
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            if re.search(edu_keywords, sentence.lower()):
                # Extract degree
                degree_match = re.search(edu_keywords, sentence.lower())
                degree = degree_match.group() if degree_match else ''
                
                # Extract year
                year_match = re.search(year_pattern, sentence)
                year = year_match.group() if year_match else ''
                
                if degree:
                    education.append({
                        'degree': degree.title(),
                        'year': year,
                        'details': sentence.strip()
                    })
        
        return education
    
    def extract_basic_info(self, text: str) -> Dict[str, str]:
        """Extract basic information from resume text."""
        try:
            return {
                'name': self.extract_name(text),
                'email': self._extract_email(text),
                'phone': self._extract_phone(text)
            }
        except Exception as e:
            self.logger.error(f"Error extracting basic info: {str(e)}")
            return {
                'name': '',
                'email': '',
                'phone': ''
            }
    
    def extract_name(self, text: str) -> str:
        """Extract name from resume text."""
        try:
            doc = nlp(text)
            
            # Look for PERSON entities in the first few sentences
            sentences = text.split('\n')[:5]  # Check first 5 lines
            
            for sentence in sentences:
                if len(sentence.strip()) > 3:
                    doc = nlp(sentence)
                    for ent in doc.ents:
                        if ent.label_ == 'PERSON' and len(ent.text.split()) <= 3:
                            # Clean the name
                            name = re.sub(r'[^\w\s]', '', ent.text).strip()
                            if len(name) > 2 and not name.lower() in ['resume', 'cv', 'curriculum']:
                                return name
            
            # Fallback: look for patterns like "Name: John Doe"
            name_patterns = [
                r'name\s*:\s*([a-zA-Z\s]+)',
                r'^([A-Z][a-z]+\s+[A-Z][a-z]+)',
                r'([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)'
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    name = match.group(1).strip()
                    if len(name) > 2:
                        return name
            
            return ''
        except Exception as e:
            self.logger.error(f"Error extracting name: {str(e)}")
            return ''
    
    def _extract_email(self, text: str) -> str:
        """Extract email from text."""
        try:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            return emails[0] if emails else ''
        except Exception as e:
            self.logger.error(f"Error extracting email: {str(e)}")
            return ''
    
    def _extract_phone(self, text: str) -> str:
        """Extract phone number from text."""
        try:
            # Various phone number patterns
            phone_patterns = [
                r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
                r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
            ]
            
            for pattern in phone_patterns:
                phones = re.findall(pattern, text)
                if phones:
                    # Clean and return the first phone number
                    phone = re.sub(r'[^\d+]', '', phones[0])
                    if len(phone) >= 10:
                        return phones[0]
            
            return ''
        except Exception as e:
            self.logger.error(f"Error extracting phone: {str(e)}")
            return ''
    
    def extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience from text."""
        experience = []
        
        # Common experience keywords
        exp_keywords = r'\b(experience|work|job|position|role|employed)\b'
        year_pattern = r'\b(19|20)\d{2}\b'
        duration_pattern = r'\b(\d+)\s*(year|month|yr|mo)s?\b'
        
        # Find experience sections
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            if re.search(exp_keywords, sentence.lower()) and len(sentence) > 20:
                # Extract years
                years = re.findall(year_pattern, sentence)
                duration = re.search(duration_pattern, sentence.lower())
                
                experience.append({
                    'description': sentence.strip(),
                    'years': years,
                    'duration': duration.group() if duration else ''
                })
        
        return experience[:5]  # Return top 5 experiences
    
    def predict_category(self, text: str, vectorizer: TfidfVectorizer, classifier: Any) -> str:
        """Predict job category from resume text."""
        try:
            # Clean and prepare text
            cleaned_text = self.clean_text(text)
            
            # Transform text using vectorizer
            text_vector = vectorizer.transform([cleaned_text])
            
            # Predict category
            category = classifier.predict(text_vector)[0]
            return category
        except Exception as e:
            self.logger.error(f"Error predicting category: {str(e)}")
            return "unknown"
        
    def predict_job_recommendation(self, text: str, vectorizer: TfidfVectorizer, classifier: Any) -> str:
        """Predict job recommendation from resume text."""
        try:
            # Clean and prepare text
            cleaned_text = self.clean_text(text)
            
            # Transform text using vectorizer
            text_vector = vectorizer.transform([cleaned_text])
            
            # Predict job recommendation
            recommendation = classifier.predict(text_vector)[0]
            return recommendation
        except Exception as e:
            self.logger.error(f"Error predicting job recommendation: {str(e)}")
            return "unknown"
    
    def match_skills_with_requirements(self, resume_skills: List[str], job_requirements: str) -> Dict[str, Any]:
        """Match resume skills with job requirements."""
        # Extract skills from job requirements text
        job_skills = self.extract_skills(job_requirements)
        
        # Also try to extract skills from comma-separated requirements
        if ',' in job_requirements:
            requirement_parts = [part.strip() for part in job_requirements.split(',')]
            for part in requirement_parts:
                # Check if this part is a skill
                part_lower = part.lower()
                if any(skill.lower() in part_lower for skill in self.skills_db):
                    job_skills.append(part)
        
        # Remove duplicates
        job_skills = list(set(job_skills))
        
        # Find matched skills (case insensitive)
        matched_skills = []
        missing_skills = []
        
        for job_skill in job_skills:
            found = False
            for resume_skill in resume_skills:
                # Check for exact match or partial match
                if (job_skill.lower() == resume_skill.lower() or 
                    job_skill.lower() in resume_skill.lower() or 
                    resume_skill.lower() in job_skill.lower()):
                    matched_skills.append(job_skill)
                    found = True
                    break
            if not found:
                missing_skills.append(job_skill)
        
        # Calculate match percentage
        total_required = len(job_skills) if job_skills else 1
        match_percentage = round((len(matched_skills) / total_required) * 100, 2)
        
        return {
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'match_percentage': match_percentage,
            'total_required_skills': total_required,
            'matched_count': len(matched_skills)
        }
        
    def calculate_match_score(self, resume_text: str, job_requirements: str) -> float:
        """Calculate match score between resume and job requirements."""
        try:
            # Clean texts
            resume_clean = self.clean_text(resume_text)
            requirements_clean = self.clean_text(job_requirements)
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([resume_clean, requirements_clean])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Scale similarity to 0-100 range
            score = round(similarity * 100, 2)
            return max(0, min(score, 100))  # Ensure score is between 0 and 100
        except Exception as e:
            self.logger.error(f"Error calculating match score: {str(e)}")
            return 0.0

# Legacy function wrappers for backward compatibility
def cleanResume(txt):
    processor = ResumeProcessor()
    return processor.clean_text(txt)

def pdf_to_text(file):
    processor = ResumeProcessor()
    return processor._extract_from_pdf(file)

def extract_contact_number_from_resume(text):
    processor = ResumeProcessor()
    return processor._extract_phone(text)

def extract_email_from_resume(text):
    processor = ResumeProcessor()
    return processor._extract_email(text)

def extract_name_from_resume(text):
    processor = ResumeProcessor()
    return processor.extract_name(text)

def extract_skills_from_resume(text):
    processor = ResumeProcessor()
    return processor.extract_skills(text)

def extract_education_from_resume(text):
    processor = ResumeProcessor()
    return processor.extract_education(text)

def predict_category(text, tfidf_vectorizer, rf_classifier):
    processor = ResumeProcessor()
    return processor.predict_category(text, tfidf_vectorizer, rf_classifier)

def job_recommendation(text, tfidf_vectorizer, rf_classifier):
    processor = ResumeProcessor()
    return processor.predict_job_recommendation(text, tfidf_vectorizer, rf_classifier)
