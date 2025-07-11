import re
import os
import json
import spacy
import docx2txt
import PyPDF2
import nltk
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer
import faiss

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

class SemanticAnalyzer:
    """
    Advanced semantic analysis using BERT/DistilBERT for better understanding
    of resume content and job requirements matching.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        try:
            # Use sentence-transformers for better semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # DistilBERT for specific NLP tasks
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            
            self.logger.info("Semantic models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading semantic models: {str(e)}")
            # Fallback to None - will use traditional methods
            self.sentence_model = None
            self.tokenizer = None
            self.bert_model = None
            
        # Skills synonyms for semantic matching
        self.skill_synonyms = {
            'javascript': ['js', 'ecmascript', 'node.js', 'nodejs'],
            'python': ['py', 'python3', 'django', 'flask'],
            'machine learning': ['ml', 'artificial intelligence', 'ai', 'deep learning'],
            'database': ['db', 'sql', 'mysql', 'postgresql', 'mongodb'],
            'web development': ['frontend', 'backend', 'full-stack', 'web dev'],
            'project management': ['pm', 'scrum master', 'agile', 'team lead'],
            'data analysis': ['analytics', 'data science', 'statistics', 'reporting'],
            'cloud computing': ['aws', 'azure', 'gcp', 'cloud services'],
            'devops': ['ci/cd', 'deployment', 'infrastructure', 'automation'],
            'leadership': ['team lead', 'management', 'supervision', 'mentoring']
        }
        
    def get_semantic_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get semantic embeddings for a list of texts using sentence transformers."""
        if self.sentence_model is None:
            return None
            
        try:
            embeddings = self.sentence_model.encode(texts)
            return embeddings
        except Exception as e:
            self.logger.error(f"Error getting embeddings: {str(e)}")
            return None
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if self.sentence_model is None:
            return 0.0
            
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
    
    def find_semantic_skills(self, text: str, skill_list: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Find skills in text using semantic similarity instead of exact matching."""
        if self.sentence_model is None:
            return []
            
        try:
            found_skills = []
            text_lower = text.lower()
            
            # Get embeddings for the entire text
            text_embedding = self.sentence_model.encode([text_lower])
            
            # Check each skill and its synonyms
            for skill in skill_list:
                skill_variants = [skill.lower()]
                
                # Add synonyms if available
                if skill.lower() in self.skill_synonyms:
                    skill_variants.extend(self.skill_synonyms[skill.lower()])
                
                max_similarity = 0.0
                best_match = skill
                
                # Check each variant
                for variant in skill_variants:
                    # Direct text search first
                    if variant in text_lower:
                        found_skills.append((skill, 1.0))
                        max_similarity = 1.0
                        break
                    
                    # Semantic similarity
                    variant_embedding = self.sentence_model.encode([variant])
                    similarity = cosine_similarity(text_embedding, variant_embedding)[0][0]
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = skill
                
                # Add skill if similarity is above threshold
                if max_similarity >= threshold and max_similarity < 1.0:
                    found_skills.append((best_match, max_similarity))
            
            # Sort by similarity score
            found_skills.sort(key=lambda x: x[1], reverse=True)
            return found_skills
            
        except Exception as e:
            self.logger.error(f"Error finding semantic skills: {str(e)}")
            return []
    
    def extract_semantic_context(self, text: str, target_skills: List[str]) -> Dict[str, Any]:
        """Extract context around skills to understand experience level and relevance."""
        if self.sentence_model is None:
            return {}
            
        try:
            sentences = nltk.sent_tokenize(text)
            context_info = {}
            
            for skill in target_skills:
                skill_contexts = []
                
                for sentence in sentences:
                    similarity = self.semantic_similarity(sentence.lower(), skill.lower())
                    if similarity > 0.3:  # Lower threshold for context
                        # Extract experience indicators
                        experience_level = self._extract_experience_level(sentence)
                        context_info[skill] = {
                            'context': sentence,
                            'similarity': similarity,
                            'experience_level': experience_level
                        }
                        skill_contexts.append({
                            'text': sentence,
                            'similarity': similarity,
                            'experience': experience_level
                        })
                
                if skill_contexts:
                    # Get the best context
                    best_context = max(skill_contexts, key=lambda x: x['similarity'])
                    context_info[skill] = best_context
            
            return context_info
            
        except Exception as e:
            self.logger.error(f"Error extracting semantic context: {str(e)}")
            return {}
    
    def _extract_experience_level(self, text: str) -> str:
        """Extract experience level indicators from text."""
        text_lower = text.lower()
        
        # Experience level indicators
        expert_indicators = ['expert', 'senior', 'lead', 'architect', 'advanced', 'extensive']
        intermediate_indicators = ['experienced', 'proficient', 'skilled', 'intermediate']
        beginner_indicators = ['beginner', 'basic', 'junior', 'entry', 'learning', 'familiar']
        
        for indicator in expert_indicators:
            if indicator in text_lower:
                return 'expert'
        
        for indicator in intermediate_indicators:
            if indicator in text_lower:
                return 'intermediate'
                
        for indicator in beginner_indicators:
            if indicator in text_lower:
                return 'beginner'
        
        # Look for years of experience
        years_match = re.search(r'(\d+)\s*(?:years?|yrs?)', text_lower)
        if years_match:
            years = int(years_match.group(1))
            if years >= 5:
                return 'expert'
            elif years >= 2:
                return 'intermediate'
            else:
                return 'beginner'
        
        return 'unknown'
    
    def semantic_job_matching(self, resume_text: str, job_requirements: str) -> Dict[str, Any]:
        """Advanced job matching using semantic understanding."""
        if self.sentence_model is None:
            return {'error': 'Semantic models not available'}
            
        try:
            # Break down job requirements into components
            job_sentences = nltk.sent_tokenize(job_requirements)
            resume_sentences = nltk.sent_tokenize(resume_text)
            
            # Get embeddings for all sentences
            all_sentences = job_sentences + resume_sentences
            embeddings = self.sentence_model.encode(all_sentences)
            
            job_embeddings = embeddings[:len(job_sentences)]
            resume_embeddings = embeddings[len(job_sentences):]
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(job_embeddings, resume_embeddings)
            
            # Find best matches for each job requirement
            matches = []
            for i, job_sentence in enumerate(job_sentences):
                best_match_idx = np.argmax(similarity_matrix[i])
                best_similarity = similarity_matrix[i][best_match_idx]
                
                if best_similarity > 0.3:  # Minimum threshold
                    matches.append({
                        'requirement': job_sentence,
                        'matched_text': resume_sentences[best_match_idx],
                        'similarity': float(best_similarity)
                    })
            
            # Calculate overall match score
            if matches:
                overall_score = np.mean([match['similarity'] for match in matches])
                coverage = len(matches) / len(job_sentences)
            else:
                overall_score = 0.0
                coverage = 0.0
            
            return {
                'overall_score': float(overall_score),
                'coverage': float(coverage),
                'detailed_matches': matches,
                'total_requirements': len(job_sentences),
                'matched_requirements': len(matches)
            }
            
        except Exception as e:
            self.logger.error(f"Error in semantic job matching: {str(e)}")
            return {'error': str(e)}

class ResumeProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize semantic analyzer
        self.semantic_analyzer = SemanticAnalyzer()
        
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
        """Extract skills from text using both traditional and semantic methods."""
        text_lower = text.lower()
        found_skills = set()
        
        # Traditional keyword-based extraction
        all_skills = []
        for category, skills in self.skills_dict.items():
            all_skills.extend(skills)
        all_skills.extend(self.skills_db)
        unique_skills = list(set(all_skills))
        
        # Traditional exact matching
        for skill in unique_skills:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                found_skills.add(skill)
        
        # Enhanced semantic skill extraction
        try:
            semantic_skills = self.semantic_analyzer.find_semantic_skills(
                text, unique_skills, threshold=0.6
            )
            
            # Add semantically found skills
            for skill, confidence in semantic_skills:
                if confidence > 0.6:  # High confidence threshold
                    found_skills.add(skill)
                    
        except Exception as e:
            self.logger.error(f"Error in semantic skill extraction: {str(e)}")
        
        return sorted(list(found_skills))
    
    def extract_skills_with_context(self, text: str) -> Dict[str, Any]:
        """Extract skills with semantic context and confidence scores."""
        try:
            # Get basic skills
            skills = self.extract_skills(text)
            
            # Get semantic context for each skill
            context_info = self.semantic_analyzer.extract_semantic_context(text, skills)
            
            # Combine traditional and semantic results
            enhanced_skills = {}
            for skill in skills:
                enhanced_skills[skill] = {
                    'found': True,
                    'confidence': 1.0,  # High confidence for exact matches
                    'context': context_info.get(skill, {}),
                    'experience_level': context_info.get(skill, {}).get('experience', 'unknown')
                }
            
            # Add semantically found skills with lower confidence
            semantic_skills = self.semantic_analyzer.find_semantic_skills(
                text, self.skills_db, threshold=0.4
            )
            
            for skill, confidence in semantic_skills:
                if skill not in enhanced_skills and confidence > 0.4:
                    enhanced_skills[skill] = {
                        'found': True,
                        'confidence': confidence,
                        'context': context_info.get(skill, {}),
                        'experience_level': context_info.get(skill, {}).get('experience', 'unknown')
                    }
            
            return enhanced_skills
            
        except Exception as e:
            self.logger.error(f"Error extracting skills with context: {str(e)}")
            return {}
        
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
        """Match resume skills with job requirements using semantic understanding."""
        # Extract skills from job requirements text (traditional method)
        job_skills = self.extract_skills(job_requirements)
        
        # Also try to extract skills from comma-separated requirements
        if ',' in job_requirements:
            requirement_parts = [part.strip() for part in job_requirements.split(',')]
            for part in requirement_parts:
                part_lower = part.lower()
                if any(skill.lower() in part_lower for skill in self.skills_db):
                    job_skills.append(part)
        
        # Remove duplicates
        job_skills = list(set(job_skills))
        
        # Traditional matching
        matched_skills = []
        missing_skills = []
        semantic_matches = []
        
        # First pass: exact and partial matching
        for job_skill in job_skills:
            found = False
            for resume_skill in resume_skills:
                if (job_skill.lower() == resume_skill.lower() or 
                    job_skill.lower() in resume_skill.lower() or 
                    resume_skill.lower() in job_skill.lower()):
                    matched_skills.append({
                        'required_skill': job_skill,
                        'matched_skill': resume_skill,
                        'match_type': 'exact',
                        'confidence': 1.0
                    })
                    found = True
                    break
            if not found:
                missing_skills.append(job_skill)
        
        # Second pass: semantic matching for missing skills
        try:
            for missing_skill in missing_skills[:]:  # Copy list to modify during iteration
                best_match = None
                best_similarity = 0.0
                
                for resume_skill in resume_skills:
                    similarity = self.semantic_analyzer.semantic_similarity(
                        missing_skill, resume_skill
                    )
                    
                    if similarity > best_similarity and similarity > 0.7:  # High threshold
                        best_similarity = similarity
                        best_match = resume_skill
                
                if best_match:
                    semantic_matches.append({
                        'required_skill': missing_skill,
                        'matched_skill': best_match,
                        'match_type': 'semantic',
                        'confidence': best_similarity
                    })
                    missing_skills.remove(missing_skill)
                    
        except Exception as e:
            self.logger.error(f"Error in semantic matching: {str(e)}")
        
        # Calculate enhanced match percentage
        total_matches = len(matched_skills) + len(semantic_matches)
        total_required = len(job_skills) if job_skills else 1
        match_percentage = round((total_matches / total_required) * 100, 2)
        
        # Calculate weighted score (exact matches worth more than semantic)
        exact_weight = 1.0
        semantic_weight = 0.8
        
        weighted_score = (
            len(matched_skills) * exact_weight + 
            sum(match['confidence'] for match in semantic_matches) * semantic_weight
        ) / total_required * 100
        
        return {
            'exact_matches': [match['required_skill'] for match in matched_skills],
            'semantic_matches': semantic_matches,
            'missing_skills': missing_skills,
            'match_percentage': match_percentage,
            'weighted_score': round(weighted_score, 2),
            'total_required_skills': total_required,
            'exact_match_count': len(matched_skills),
            'semantic_match_count': len(semantic_matches),
            'all_matches': matched_skills + semantic_matches
        }
        
    def calculate_match_score(self, resume_text: str, job_requirements: str, 
                            job_category: str = None, category_description: str = None,
                            required_experience: str = None) -> Dict[str, Any]:
        """Calculate comprehensive match score using multiple weighted components."""
        try:
            # Traditional TF-IDF approach
            resume_clean = self.clean_text(resume_text)
            requirements_clean = self.clean_text(job_requirements)
            
            # Include category description in analysis if provided
            enhanced_requirements = requirements_clean
            if category_description:
                enhanced_requirements += " " + self.clean_text(category_description)
            
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([resume_clean, enhanced_requirements])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Semantic similarity
            semantic_score = 0.0
            semantic_details = {}
            
            try:
                semantic_score = self.semantic_analyzer.semantic_similarity(
                    resume_text, job_requirements + " " + (category_description or "")
                )
                
                # Get detailed semantic matching
                semantic_details = self.semantic_analyzer.semantic_job_matching(
                    resume_text, job_requirements
                )
                
            except Exception as e:
                self.logger.error(f"Error in semantic scoring: {str(e)}")
                semantic_details = {'error': str(e)}
            
            # Skills-based matching
            resume_skills = self.extract_skills(resume_text)
            skills_match = self.match_skills_with_requirements(resume_skills, job_requirements)
            
            # Experience level matching
            experience_score = 50.0  # Default neutral score
            if required_experience:
                experience_score = self.calculate_experience_match_score(resume_text, required_experience)
            
            # Category relevance (if category provided)
            category_score = 50.0  # Default neutral
            if job_category:
                category_score = self.semantic_analyzer.semantic_similarity(
                    resume_text, job_category + " " + (category_description or "")
                ) * 100
            
            # Dynamic scoring weights based on availability of components
            base_weights = {
                'tfidf': 0.25,
                'semantic': 0.35,
                'skills': 0.25,
                'experience': 0.10,
                'category': 0.05
            }
            
            # Adjust weights if some components are missing
            active_weights = {}
            total_weight = 0
            
            for component, weight in base_weights.items():
                if component == 'experience' and not required_experience:
                    continue
                if component == 'category' and not job_category:
                    continue
                active_weights[component] = weight
                total_weight += weight
            
            # Normalize weights to sum to 1.0
            for component in active_weights:
                active_weights[component] /= total_weight
            
            # Normalize scores to 0-100 range
            tfidf_score = tfidf_similarity * 100
            semantic_score_normalized = semantic_score * 100
            skills_score = skills_match.get('weighted_score', 0)
            
            # Calculate weighted final score
            final_score = (
                tfidf_score * active_weights.get('tfidf', 0) +
                semantic_score_normalized * active_weights.get('semantic', 0) +
                skills_score * active_weights.get('skills', 0) +
                experience_score * active_weights.get('experience', 0) +
                category_score * active_weights.get('category', 0)
            )
            
            # Ensure score is between 0 and 100
            final_score = max(0, min(final_score, 100))
            
            return {
                'final_score': round(final_score, 2),
                'component_scores': {
                    'tfidf_score': round(tfidf_score, 2),
                    'semantic_score': round(semantic_score_normalized, 2),
                    'skills_score': round(skills_score, 2),
                    'experience_score': round(experience_score, 2),
                    'category_score': round(category_score, 2)
                },
                'skills_analysis': skills_match,
                'semantic_details': semantic_details,
                'weights_used': active_weights,
                'match_breakdown': {
                    'skills_matched': skills_match.get('exact_match_count', 0) + skills_match.get('semantic_match_count', 0),
                    'skills_total': skills_match.get('total_required_skills', 0),
                    'experience_alignment': round(experience_score, 1),
                    'category_relevance': round(category_score, 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive match score: {str(e)}")
            return {
                'final_score': 0.0,
                'error': str(e),
                'component_scores': {
                    'tfidf_score': 0.0,
                    'semantic_score': 0.0,
                    'skills_score': 0.0,
                    'experience_score': 0.0,
                    'category_score': 0.0
                }
            }
    
    def calculate_experience_match_score(self, resume_text: str, required_experience: str) -> float:
        """Calculate how well candidate's experience matches required experience level."""
        try:
            resume_seniority = self._determine_seniority_level(resume_text)
            required_seniority = required_experience.lower()
            
            # Experience level mapping
            level_scores = {
                ('junior', 'entry'): 1.0, ('junior', 'junior'): 1.0,
                ('junior', 'mid-level'): 0.6, ('junior', 'senior'): 0.2,
                ('mid-level', 'entry'): 0.8, ('mid-level', 'junior'): 0.8,
                ('mid-level', 'mid-level'): 1.0, ('mid-level', 'senior'): 0.6,
                ('senior', 'entry'): 0.9, ('senior', 'junior'): 0.9,
                ('senior', 'mid-level'): 0.9, ('senior', 'senior'): 1.0
            }
            
            # Map common variations
            experience_mapping = {
                'entry level': 'entry', 'entry-level': 'entry',
                'mid level': 'mid-level', 'intermediate': 'mid-level',
                'experienced': 'mid-level', 'senior level': 'senior'
            }
            
            mapped_required = experience_mapping.get(required_seniority, required_seniority)
            
            score = level_scores.get((resume_seniority, mapped_required), 0.5)
            return score * 100  # Convert to percentage
            
        except Exception as e:
            self.logger.error(f"Error calculating experience match: {str(e)}")
            return 50.0  # Neutral score on error
    
    def analyze_transferable_skills(self, resume_text: str, target_domain: str) -> Dict[str, Any]:
        """Analyze transferable skills for career transitions using semantic understanding."""
        try:
            # Extract all skills with context
            skills_with_context = self.extract_skills_with_context(resume_text)
            
            # Define transferable skill categories
            transferable_categories = {
                'leadership': ['management', 'team lead', 'supervision', 'mentoring', 'project management'],
                'communication': ['presentation', 'writing', 'documentation', 'client relations'],
                'analytical': ['problem solving', 'data analysis', 'research', 'critical thinking'],
                'technical': ['programming', 'database', 'web development', 'software'],
                'creative': ['design', 'innovation', 'creativity', 'user experience']
            }
            
            transferable_analysis = {}
            
            for category, keywords in transferable_categories.items():
                category_skills = []
                
                for skill, details in skills_with_context.items():
                    # Check if skill belongs to this category using semantic similarity
                    for keyword in keywords:
                        similarity = self.semantic_analyzer.semantic_similarity(skill, keyword)
                        if similarity > 0.5:
                            category_skills.append({
                                'skill': skill,
                                'relevance': similarity,
                                'experience_level': details.get('experience_level', 'unknown'),
                                'confidence': details.get('confidence', 0.0)
                            })
                            break
                
                if category_skills:
                    transferable_analysis[category] = {
                        'skills': category_skills,
                        'strength': len(category_skills),
                        'avg_relevance': np.mean([s['relevance'] for s in category_skills])
                    }
            
            # Calculate transferability to target domain
            domain_relevance = 0.0
            if target_domain:
                domain_similarity = self.semantic_analyzer.semantic_similarity(
                    resume_text, target_domain
                )
                domain_relevance = domain_similarity
            
            return {
                'transferable_skills': transferable_analysis,
                'domain_relevance': float(domain_relevance),
                'total_transferable_count': sum(
                    len(cat['skills']) for cat in transferable_analysis.values()
                ),
                'strongest_category': max(
                    transferable_analysis.keys(),
                    key=lambda k: transferable_analysis[k]['avg_relevance']
                ) if transferable_analysis else None
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing transferable skills: {str(e)}")
            return {'error': str(e)}
    
    def semantic_resume_summary(self, resume_text: str) -> Dict[str, Any]:
        """Generate a comprehensive semantic summary of the resume."""
        try:
            # Extract various components
            basic_info = self.extract_basic_info(resume_text)
            skills = self.extract_skills_with_context(resume_text)
            experience = self.extract_experience(resume_text)
            education = self.extract_education(resume_text)
            
            # Analyze skill diversity and depth
            skill_categories = {}
            for skill, details in skills.items():
                # Categorize skills semantically
                for category, category_skills in self.skills_dict.items():
                    for cat_skill in category_skills:
                        similarity = self.semantic_analyzer.semantic_similarity(skill, cat_skill)
                        if similarity > 0.6:
                            if category not in skill_categories:
                                skill_categories[category] = []
                            skill_categories[category].append({
                                'skill': skill,
                                'similarity': similarity,
                                'experience_level': details.get('experience_level', 'unknown')
                            })
                            break
            
            # Calculate experience depth
            experience_indicators = {
                'senior_keywords': ['senior', 'lead', 'manager', 'director', 'architect'],
                'years_mentioned': re.findall(r'(\d+)\s*(?:years?|yrs?)', resume_text.lower()),
                'companies_count': len(re.findall(r'(?:company|corp|inc|ltd|llc)', resume_text.lower()))
            }
            
            return {
                'basic_info': basic_info,
                'skill_summary': {
                    'total_skills': len(skills),
                    'categorized_skills': skill_categories,
                    'skill_diversity': len(skill_categories),
                    'avg_confidence': np.mean([s.get('confidence', 0) for s in skills.values()]) if skills else 0
                },
                'experience_summary': {
                    'total_experiences': len(experience),
                    'experience_indicators': experience_indicators,
                    'seniority_level': self._determine_seniority_level(resume_text)
                },
                'education_summary': {
                    'education_count': len(education),
                    'education_details': education
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating semantic summary: {str(e)}")
            return {'error': str(e)}
    
    def _determine_seniority_level(self, text: str) -> str:
        """Determine seniority level based on semantic analysis."""
        text_lower = text.lower()
        
        senior_indicators = [
            'senior', 'lead', 'principal', 'architect', 'manager', 'director',
            'head of', 'chief', 'vp', 'vice president'
        ]
        
        mid_indicators = [
            'developer', 'engineer', 'analyst', 'specialist', 'consultant'
        ]
        
        junior_indicators = [
            'junior', 'entry', 'associate', 'intern', 'trainee', 'assistant'
        ]
        
        # Count indicators
        senior_count = sum(1 for indicator in senior_indicators if indicator in text_lower)
        mid_count = sum(1 for indicator in mid_indicators if indicator in text_lower)
        junior_count = sum(1 for indicator in junior_indicators if indicator in text_lower)
        
        # Check years of experience
        years_matches = re.findall(r'(\d+)\s*(?:years?|yrs?)', text_lower)
        max_years = max([int(year) for year in years_matches]) if years_matches else 0
        
        # Determine level
        if senior_count > 0 or max_years >= 7:
            return 'senior'
        elif junior_count > 0 or max_years <= 2:
            return 'junior'
        else:
            return 'mid-level'
        
    def calculate_priority_weighted_skills_score(self, resume_skills: List[str], 
                                                job_requirements: str, 
                                                priority_skills: List[str] = None) -> Dict[str, Any]:
        """Calculate skills score with priority weighting for critical skills."""
        try:
            # Get basic skills matching
            basic_match = self.match_skills_with_requirements(resume_skills, job_requirements)
            
            if not priority_skills:
                return basic_match
            
            # Calculate priority skills matching
            priority_matched = 0
            priority_total = len(priority_skills)
            priority_details = []
            
            for priority_skill in priority_skills:
                skill_found = False
                best_match = None
                best_similarity = 0.0
                
                # Check exact matches first
                for resume_skill in resume_skills:
                    if (priority_skill.lower() == resume_skill.lower() or 
                        priority_skill.lower() in resume_skill.lower() or 
                        resume_skill.lower() in priority_skill.lower()):
                        priority_matched += 1
                        skill_found = True
                        priority_details.append({
                            'required': priority_skill,
                            'matched': resume_skill,
                            'type': 'exact',
                            'score': 1.0
                        })
                        break
                
                # If no exact match, try semantic matching
                if not skill_found:
                    for resume_skill in resume_skills:
                        similarity = self.semantic_analyzer.semantic_similarity(
                            priority_skill, resume_skill
                        )
                        if similarity > best_similarity and similarity > 0.75:  # High threshold for priority
                            best_similarity = similarity
                            best_match = resume_skill
                    
                    if best_match:
                        priority_matched += best_similarity  # Partial credit for semantic match
                        priority_details.append({
                            'required': priority_skill,
                            'matched': best_match,
                            'type': 'semantic',
                            'score': best_similarity
                        })
            
            # Calculate enhanced score with priority weighting
            priority_score = (priority_matched / priority_total * 100) if priority_total > 0 else 100
            
            # Combine with basic score (70% basic, 30% priority)
            enhanced_score = (basic_match.get('weighted_score', 0) * 0.7 + priority_score * 0.3)
            
            return {
                **basic_match,  # Include all basic match data
                'priority_score': round(priority_score, 2),
                'enhanced_weighted_score': round(enhanced_score, 2),
                'priority_details': priority_details,
                'priority_match_rate': round(priority_matched / priority_total * 100, 2) if priority_total > 0 else 100
            }
            
        except Exception as e:
            self.logger.error(f"Error in priority weighted scoring: {str(e)}")
            return self.match_skills_with_requirements(resume_skills, job_requirements)
        
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
