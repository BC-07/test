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
            # Temporarily disabled to avoid network issues during testing
            # self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.sentence_model = None  # Temporary disable
            
            # DistilBERT for specific NLP tasks
            # self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
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

class BiasDetector:
    """
    Bias detection and mitigation system for fair candidate evaluation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Bias indicators by category
        self.bias_indicators = {
            'gender': {
                'male_indicators': [
                    'he', 'his', 'him', 'mr', 'mister', 'gentleman', 'guy', 'man', 'male',
                    'father', 'husband', 'son', 'brother', 'uncle', 'nephew'
                ],
                'female_indicators': [
                    'she', 'her', 'hers', 'ms', 'mrs', 'miss', 'lady', 'woman', 'female',
                    'mother', 'wife', 'daughter', 'sister', 'aunt', 'niece'
                ],
                'biased_terms': [
                    'girl', 'girls', 'boys', 'guys', 'gal', 'chick', 'dude', 'bro'
                ]
            },
            'age': {
                'young_indicators': [
                    'young', 'recent graduate', 'fresh graduate', 'new graduate',
                    'millennial', 'gen z', 'digital native', 'energetic', 'dynamic'
                ],
                'mature_indicators': [
                    'experienced', 'seasoned', 'veteran', 'mature', 'senior professional',
                    'established', 'traditional', 'old school'
                ],
                'biased_terms': [
                    'old', 'elderly', 'aging', 'outdated', 'obsolete', 'legacy mindset'
                ]
            },
            'education': {
                'elite_indicators': [
                    'harvard', 'stanford', 'mit', 'yale', 'princeton', 'oxford', 'cambridge',
                    'ivy league', 'top tier', 'prestigious', 'elite university'
                ],
                'non_traditional': [
                    'community college', 'self-taught', 'bootcamp', 'online course',
                    'certification', 'autodidact', 'non-degree'
                ]
            },
            'socioeconomic': {
                'privilege_indicators': [
                    'private school', 'boarding school', 'country club', 'yacht club',
                    'trust fund', 'family business', 'legacy', 'inherited'
                ],
                'disadvantage_indicators': [
                    'scholarship', 'work-study', 'part-time job', 'financial aid',
                    'first generation', 'immigrant', 'refugee'
                ]
            },
            'name_bias': {
                'western_names': [
                    'john', 'michael', 'david', 'james', 'robert', 'william',
                    'mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'barbara'
                ],
                'ethnic_patterns': {
                    'hispanic': ['rodriguez', 'garcia', 'martinez', 'lopez', 'gonzalez'],
                    'asian': ['chen', 'wang', 'li', 'zhang', 'kim', 'patel', 'singh'],
                    'african': ['washington', 'jefferson', 'williams', 'jackson', 'johnson'],
                    'middle_eastern': ['ahmed', 'hassan', 'ali', 'omar', 'fatima']
                }
            }
        }
        
        # Fair evaluation weights to counteract bias
        self.fair_weights = {
            'skills_relevance': 0.40,      # Focus on actual skills
            'experience_quality': 0.30,    # Quality over quantity
            'problem_solving': 0.15,       # Analytical abilities
            'cultural_add': 0.10,          # Diversity as strength
            'growth_potential': 0.05       # Learning ability
        }
    
    def detect_bias_indicators(self, resume_text: str, candidate_name: str = '') -> Dict[str, Any]:
        """Detect potential bias indicators in resume evaluation."""
        try:
            bias_detected = {
                'gender_bias': self._detect_gender_bias(resume_text),
                'age_bias': self._detect_age_bias(resume_text),
                'education_bias': self._detect_education_bias(resume_text),
                'name_bias': self._detect_name_bias(candidate_name),
                'socioeconomic_bias': self._detect_socioeconomic_bias(resume_text),
                'overall_risk': 'low'
            }
            
            # Calculate overall bias risk
            risk_factors = sum(1 for category in bias_detected.values() 
                             if isinstance(category, dict) and category.get('risk_level') in ['medium', 'high'])
            
            if risk_factors >= 3:
                bias_detected['overall_risk'] = 'high'
            elif risk_factors >= 1:
                bias_detected['overall_risk'] = 'medium'
            
            return bias_detected
            
        except Exception as e:
            self.logger.error(f"Error detecting bias indicators: {str(e)}")
            return {'error': str(e), 'overall_risk': 'unknown'}
    
    def _detect_gender_bias(self, text: str) -> Dict[str, Any]:
        """Detect gender bias indicators."""
        text_lower = text.lower()
        
        male_count = sum(1 for term in self.bias_indicators['gender']['male_indicators'] 
                        if term in text_lower)
        female_count = sum(1 for term in self.bias_indicators['gender']['female_indicators'] 
                          if term in text_lower)
        biased_terms = [term for term in self.bias_indicators['gender']['biased_terms'] 
                       if term in text_lower]
        
        # Determine bias risk
        total_indicators = male_count + female_count + len(biased_terms)
        if total_indicators > 5 or len(biased_terms) > 0:
            risk = 'high'
        elif total_indicators > 2:
            risk = 'medium'
        else:
            risk = 'low'
        
        return {
            'male_indicators': male_count,
            'female_indicators': female_count,
            'biased_terms_found': biased_terms,
            'risk_level': risk,
            'mitigation_needed': risk in ['medium', 'high']
        }
    
    def _detect_age_bias(self, text: str) -> Dict[str, Any]:
        """Detect age bias indicators."""
        text_lower = text.lower()
        
        young_indicators = sum(1 for term in self.bias_indicators['age']['young_indicators'] 
                              if term in text_lower)
        mature_indicators = sum(1 for term in self.bias_indicators['age']['mature_indicators'] 
                               if term in text_lower)
        biased_terms = [term for term in self.bias_indicators['age']['biased_terms'] 
                       if term in text_lower]
        
        # Check graduation dates for age inference
        grad_years = re.findall(r'\b(19|20)\d{2}\b', text)
        current_year = 2025
        inferred_ages = [current_year - int(year) for year in grad_years if len(year) == 4]
        
        risk = 'low'
        if len(biased_terms) > 0:
            risk = 'high'
        elif young_indicators > 2 or mature_indicators > 2:
            risk = 'medium'
        elif inferred_ages and (min(inferred_ages) > 30 or max(inferred_ages) < 5):
            risk = 'medium'
        
        return {
            'young_indicators': young_indicators,
            'mature_indicators': mature_indicators,
            'biased_terms_found': biased_terms,
            'inferred_experience_years': inferred_ages,
            'risk_level': risk,
            'mitigation_needed': risk in ['medium', 'high']
        }
    
    def _detect_education_bias(self, text: str) -> Dict[str, Any]:
        """Detect education bias indicators."""
        text_lower = text.lower()
        
        elite_count = sum(1 for term in self.bias_indicators['education']['elite_indicators'] 
                         if term in text_lower)
        non_traditional = sum(1 for term in self.bias_indicators['education']['non_traditional'] 
                             if term in text_lower)
        
        risk = 'low'
        if elite_count > 0:
            risk = 'medium'  # Elite education shouldn't be the primary factor
        elif non_traditional > 2:
            risk = 'medium'  # Ensure non-traditional education isn't penalized
        
        return {
            'elite_education_indicators': elite_count,
            'non_traditional_education': non_traditional,
            'risk_level': risk,
            'mitigation_needed': risk in ['medium', 'high']
        }
    
    def _detect_name_bias(self, name: str) -> Dict[str, Any]:
        """Detect potential name-based bias."""
        if not name:
            return {'risk_level': 'low', 'mitigation_needed': False}
        
        name_lower = name.lower()
        detected_patterns = {}
        
        # Check for ethnic name patterns
        for ethnicity, patterns in self.bias_indicators['name_bias']['ethnic_patterns'].items():
            matches = sum(1 for pattern in patterns if pattern in name_lower)
            if matches > 0:
                detected_patterns[ethnicity] = matches
        
        # Check against Western names
        western_match = sum(1 for western_name in self.bias_indicators['name_bias']['western_names'] 
                           if western_name in name_lower)
        
        risk = 'medium' if detected_patterns else 'low'
        
        return {
            'ethnic_patterns_detected': detected_patterns,
            'western_name_similarity': western_match > 0,
            'risk_level': risk,
            'mitigation_needed': risk in ['medium', 'high']
        }
    
    def _detect_socioeconomic_bias(self, text: str) -> Dict[str, Any]:
        """Detect socioeconomic bias indicators."""
        text_lower = text.lower()
        
        privilege_indicators = sum(1 for term in self.bias_indicators['socioeconomic']['privilege_indicators'] 
                                  if term in text_lower)
        disadvantage_indicators = sum(1 for term in self.bias_indicators['socioeconomic']['disadvantage_indicators'] 
                                     if term in text_lower)
        
        risk = 'low'
        if privilege_indicators > 0 or disadvantage_indicators > 1:
            risk = 'medium'
        
        return {
            'privilege_indicators': privilege_indicators,
            'disadvantage_indicators': disadvantage_indicators,
            'risk_level': risk,
            'mitigation_needed': risk in ['medium', 'high']
        }
    
    def apply_bias_mitigation(self, original_score: float, bias_analysis: Dict[str, Any], 
                            skills_match: Dict[str, Any], resume_text: str) -> Dict[str, Any]:
        """Apply bias mitigation techniques to adjust scoring."""
        try:
            mitigation_applied = []
            adjusted_score = original_score
            
            # Focus on skills-based evaluation
            skills_weight_boost = 0.0
            if bias_analysis.get('overall_risk') in ['medium', 'high']:
                skills_weight_boost = 0.15  # Boost skills importance
                mitigation_applied.append('skills_focus_enhancement')
            
            # Name bias mitigation
            name_bias = bias_analysis.get('name_bias', {})
            if name_bias.get('mitigation_needed', False):
                # Apply slight positive adjustment for non-Western names
                if name_bias.get('ethnic_patterns_detected'):
                    adjusted_score += 2.0  # Small positive bias correction
                    mitigation_applied.append('name_bias_correction')
            
            # Education bias mitigation
            edu_bias = bias_analysis.get('education_bias', {})
            if edu_bias.get('mitigation_needed', False):
                if edu_bias.get('non_traditional_education', 0) > 0:
                    adjusted_score += 3.0  # Boost for non-traditional education
                    mitigation_applied.append('education_diversity_bonus')
            
            # Age bias mitigation - focus on skills over years
            age_bias = bias_analysis.get('age_bias', {})
            if age_bias.get('mitigation_needed', False):
                # Reduce weight of experience, increase weight of skills
                skills_score = skills_match.get('weighted_score', 0)
                if skills_score > 70:  # High skills compensate for age bias
                    adjusted_score += 1.5
                    mitigation_applied.append('age_skills_compensation')
            
            # Gender bias mitigation
            gender_bias = bias_analysis.get('gender_bias', {})
            if gender_bias.get('mitigation_needed', False):
                # Apply gender-neutral evaluation boost
                adjusted_score += 1.0
                mitigation_applied.append('gender_neutral_evaluation')
            
            # Ensure score doesn't exceed 100
            adjusted_score = min(adjusted_score, 100.0)
            
            # Calculate fairness metrics
            fairness_score = self._calculate_fairness_score(bias_analysis, skills_match)
            
            return {
                'original_score': original_score,
                'adjusted_score': round(adjusted_score, 2),
                'bias_adjustment': round(adjusted_score - original_score, 2),
                'mitigation_applied': mitigation_applied,
                'fairness_score': fairness_score,
                'explanation': self._generate_fairness_explanation(mitigation_applied, bias_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error applying bias mitigation: {str(e)}")
            return {
                'original_score': original_score,
                'adjusted_score': original_score,
                'error': str(e)
            }
    
    def _calculate_fairness_score(self, bias_analysis: Dict[str, Any], skills_match: Dict[str, Any]) -> float:
        """Calculate a fairness score for the evaluation."""
        fairness_factors = []
        
        # Skills-based evaluation strength
        skills_score = skills_match.get('weighted_score', 0)
        skills_fairness = min(skills_score / 80.0, 1.0)  # Normalize to 0-1
        fairness_factors.append(skills_fairness * 0.5)
        
        # Bias risk inverse (lower bias = higher fairness)
        risk_scores = {'low': 1.0, 'medium': 0.6, 'high': 0.3}
        overall_risk = bias_analysis.get('overall_risk', 'low')
        bias_fairness = risk_scores.get(overall_risk, 0.5)
        fairness_factors.append(bias_fairness * 0.3)
        
        # Evaluation diversity (multiple factors considered)
        diversity_score = 0.8  # Base diversity score
        fairness_factors.append(diversity_score * 0.2)
        
        return round(sum(fairness_factors) * 100, 1)
    
    def _generate_fairness_explanation(self, mitigation_applied: List[str], bias_analysis: Dict[str, Any]) -> str:
        """Generate explanation of fairness measures applied."""
        if not mitigation_applied:
            return "No bias indicators detected. Standard evaluation applied."
        
        explanations = {
            'skills_focus_enhancement': "Increased focus on skills-based evaluation",
            'name_bias_correction': "Applied name-neutral evaluation adjustment",
            'education_diversity_bonus': "Enhanced scoring for diverse educational backgrounds",
            'age_skills_compensation': "Prioritized skills over experience length",
            'gender_neutral_evaluation': "Applied gender-neutral evaluation standards"
        }
        
        applied_explanations = [explanations.get(action, action) for action in mitigation_applied]
        return "Fairness measures applied: " + "; ".join(applied_explanations)

class PersonalDataSheetProcessor:
    """
    Specialized processor for Personal Data Sheets (PDS) with different scoring criteria
    and evaluation methods compared to traditional resumes.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize semantic analyzer and bias detector
        self.semantic_analyzer = SemanticAnalyzer()
        self.bias_detector = BiasDetector()
        
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
        self.pds_scoring_criteria = {
            'education': {
                'weight': 0.25,
                'subcriteria': {
                    'relevance': 0.4,
                    'level': 0.3,
                    'institution': 0.2,
                    'grades': 0.1
                }
            },
            'experience': {
                'weight': 0.30,
                'subcriteria': {
                    'relevance': 0.5,
                    'duration': 0.3,
                    'responsibilities': 0.2
                }
            },
            'skills': {
                'weight': 0.20,
                'subcriteria': {
                    'technical_match': 0.6,
                    'certifications': 0.4
                }
            },
            'personal_attributes': {
                'weight': 0.15,
                'subcriteria': {
                    'eligibility': 0.5,
                    'awards': 0.3,
                    'training': 0.2
                }
            },
            'additional_qualifications': {
                'weight': 0.10,
                'subcriteria': {
                    'languages': 0.4,
                    'licenses': 0.3,
                    'volunteer_work': 0.3
                }
            }
        }
    
    def extract_basic_info(self, content):
        """Extract basic personal information from PDS content."""
        info = {}
        content_lower = content.lower()
        
        # Extract name patterns
        name_patterns = [
            r'name[:\s]+([^\n\r]+)',
            r'surname[:\s]+([^\n\r]+)',
            r'first name[:\s]+([^\n\r]+)',
            r'full name[:\s]+([^\n\r]+)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, content_lower)
            if match:
                info['name'] = match.group(1).strip().title()
                break
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, content)
        if email_match:
            info['email'] = email_match.group()
        
        # Extract phone
        phone_patterns = [
            r'phone[:\s]*([0-9\-\+\(\)\s]+)',
            r'mobile[:\s]*([0-9\-\+\(\)\s]+)',
            r'contact[:\s]*([0-9\-\+\(\)\s]+)'
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, content_lower)
            if match:
                phone = re.sub(r'[^\d\+]', '', match.group(1))
                if len(phone) >= 10:
                    info['phone'] = phone
                    break
        
        return info

    def extract_education_pds(self, content):
        """Extract education information specifically from PDS format."""
        education = []
        content_lower = content.lower()
        
        # PDS-specific education patterns
        education_section_pattern = r'educational\s+background.*?(?=work\s+experience|experience|voluntary|learning|other\s+information|$)'
        education_section_match = re.search(education_section_pattern, content_lower, re.DOTALL)
        
        if education_section_match:
            education_content = education_section_match.group()
            
            # Extract degree, school, and year patterns
            degree_patterns = [
                r'(bachelor[^/\n]*)',
                r'(master[^/\n]*)',
                r'(doctorate[^/\n]*)',
                r'(phd[^/\n]*)',
                r'(college[^/\n]*)',
                r'(university[^/\n]*)',
                r'(diploma[^/\n]*)',
                r'(certificate[^/\n]*)'
            ]
            
            for pattern in degree_patterns:
                matches = re.findall(pattern, education_content)
                for match in matches:
                    if len(match.strip()) > 3:  # Filter out very short matches
                        education.append(match.strip().title())
        
        return education if education else ['Education information not clearly specified']

    def extract_experience_pds(self, content):
        """Extract work experience specifically from PDS format."""
        experience = []
        content_lower = content.lower()
        
        # PDS-specific experience patterns
        experience_section_pattern = r'work\s+experience.*?(?=voluntary|learning|other\s+information|references|$)'
        experience_section_match = re.search(experience_section_pattern, content_lower, re.DOTALL)
        
        if experience_section_match:
            experience_content = experience_section_match.group()
            
            # Extract position and company patterns
            position_patterns = [
                r'position[:\s]*([^\n\r]+)',
                r'job\s+title[:\s]*([^\n\r]+)',
                r'designation[:\s]*([^\n\r]+)'
            ]
            
            for pattern in position_patterns:
                matches = re.findall(pattern, experience_content)
                for match in matches:
                    if len(match.strip()) > 3:
                        experience.append(match.strip().title())
        
        return experience if experience else ['Work experience not clearly specified']

    def extract_skills_pds(self, content):
        """Extract skills specifically from PDS format."""
        skills = []
        content_lower = content.lower()
        
        # PDS-specific skills sections
        skills_section_patterns = [
            r'special\s+skills.*?(?=hobbies|membership|references|$)',
            r'other\s+information.*?(?=references|signature|$)',
            r'skills.*?(?=hobbies|membership|references|$)'
        ]
        
        for pattern in skills_section_patterns:
            section_match = re.search(pattern, content_lower, re.DOTALL)
            if section_match:
                skills_content = section_match.group()
                
                # Extract individual skills
                for category, skill_list in self.skills_dict.items():
                    for skill in skill_list:
                        if skill.lower() in skills_content:
                            skills.append(skill.title())
        
        return list(set(skills)) if skills else ['Skills not clearly specified']

    def extract_skills(self, text):
        """Extract skills using both traditional and semantic methods."""
        try:
            skills = set()
            text_lower = text.lower()
            
            # Traditional keyword matching
            for category, skills_list in self.skills_dict.items():
                for skill in skills_list:
                    if skill.lower() in text_lower:
                        skills.add(skill)
            
            # Semantic skill extraction
            semantic_skills = self.semantic_analyzer.find_semantic_skills(
                text, list(skills)
            )
            skills.update(semantic_skills)
            
            return list(skills)
            
        except Exception as e:
            self.logger.error(f"Error in semantic skill extraction: {str(e)}")
            # Fallback to traditional method
            return self.extract_skills_traditional(text)

    def extract_skills_traditional(self, text):
        """Traditional keyword-based skill extraction."""
        skills = set()
        text_lower = text.lower()
        
        for category, skills_list in self.skills_dict.items():
            for skill in skills_list:
                if skill.lower() in text_lower:
                    skills.add(skill)
        
        return list(skills)
    
    def extract_pds_data(self, file_path):
        """Extract PDS data from Excel file - main extraction method"""
        try:
            import pandas as pd
            
            logger.info(f"🔍 Extracting PDS data from Excel file: {file_path}")
            
            # Try to read different sheet names commonly used in PDS files
            sheet_names = ['C1', 'Sheet1', 'PDS', 'PersonalDataSheet']
            df = None
            
            for sheet in sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet, header=None)
                    logger.info(f"✅ Successfully read sheet: {sheet}")
                    break
                except:
                    continue
            
            if df is None:
                # Try reading without specifying sheet name
                df = pd.read_excel(file_path, header=None)
            
            # Extract structured data from Excel
            extracted_data = {
                'educational_background': self._extract_education_from_excel(df),
                'work_experience': self._extract_work_experience_from_excel(df),
                'learning_development': self._extract_training_from_excel(df),
                'civil_service_eligibility': self._extract_eligibility_from_excel(df),
                'voluntary_work': self._extract_voluntary_work_from_excel(df)
            }
            
            logger.info(f"✅ PDS extraction completed successfully")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting PDS data: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    def _extract_education_from_excel(self, df):
        """Extract educational background from Excel DataFrame"""
        education_data = []
        try:
            # Look for education section markers
            for idx, row in df.iterrows():
                row_str = ' '.join([str(cell) for cell in row if pd.notna(cell)]).upper()
                
                if any(marker in row_str for marker in ['EDUCATIONAL BACKGROUND', 'EDUCATION', 'TERTIARY', 'SECONDARY', 'PRIMARY']):
                    # Found education section, extract data from following rows
                    for i in range(idx + 1, min(idx + 10, len(df))):
                        edu_row = df.iloc[i]
                        edu_values = [str(cell) for cell in edu_row if pd.notna(cell) and str(cell).strip() != '']
                        
                        if len(edu_values) >= 3:
                            education_data.append({
                                'level': edu_values[0] if len(edu_values) > 0 else 'N/A',
                                'school': edu_values[1] if len(edu_values) > 1 else 'N/A',
                                'degree_course': edu_values[2] if len(edu_values) > 2 else 'N/A',
                                'year_graduated': edu_values[3] if len(edu_values) > 3 else 'N/A',
                                'honors': edu_values[4] if len(edu_values) > 4 else 'N/A'
                            })
                    break
        except Exception as e:
            logger.warning(f"Error extracting education: {e}")
        
        return education_data

    def _extract_work_experience_from_excel(self, df):
        """Extract work experience from Excel DataFrame"""
        experience_data = []
        try:
            # Look for work experience section
            for idx, row in df.iterrows():
                row_str = ' '.join([str(cell) for cell in row if pd.notna(cell)]).upper()
                
                if any(marker in row_str for marker in ['WORK EXPERIENCE', 'EMPLOYMENT', 'POSITION', 'COMPANY']):
                    # Extract work experience data
                    for i in range(idx + 1, min(idx + 15, len(df))):
                        exp_row = df.iloc[i]
                        exp_values = [str(cell) for cell in exp_row if pd.notna(cell) and str(cell).strip() != '']
                        
                        if len(exp_values) >= 3:
                            experience_data.append({
                                'position': exp_values[0] if len(exp_values) > 0 else 'N/A',
                                'company': exp_values[1] if len(exp_values) > 1 else 'N/A',
                                'date_from': exp_values[2] if len(exp_values) > 2 else 'N/A',
                                'date_to': exp_values[3] if len(exp_values) > 3 else 'N/A',
                                'salary': exp_values[4] if len(exp_values) > 4 else 'N/A',
                                'grade': exp_values[5] if len(exp_values) > 5 else 'N/A'
                            })
                    break
        except Exception as e:
            logger.warning(f"Error extracting work experience: {e}")
        
        return experience_data

    def _extract_training_from_excel(self, df):
        """Extract training and development from Excel DataFrame"""
        training_data = []
        try:
            # Look for training/learning development section
            for idx, row in df.iterrows():
                row_str = ' '.join([str(cell) for cell in row if pd.notna(cell)]).upper()
                
                if any(marker in row_str for marker in ['LEARNING AND DEVELOPMENT', 'TRAINING', 'SEMINAR', 'WORKSHOP']):
                    # Extract training data
                    for i in range(idx + 1, min(idx + 20, len(df))):
                        train_row = df.iloc[i]
                        train_values = [str(cell) for cell in train_row if pd.notna(cell) and str(cell).strip() != '']
                        
                        if len(train_values) >= 2:
                            hours = 0
                            try:
                                hours = float(train_values[2]) if len(train_values) > 2 else 0
                            except:
                                hours = 0
                            
                            training_data.append({
                                'title': train_values[0] if len(train_values) > 0 else 'N/A',
                                'conductor': train_values[1] if len(train_values) > 1 else 'N/A',
                                'hours': hours,
                                'type': train_values[3] if len(train_values) > 3 else 'N/A'
                            })
                    break
        except Exception as e:
            logger.warning(f"Error extracting training: {e}")
        
        return training_data

    def _extract_eligibility_from_excel(self, df):
        """Extract civil service eligibility from Excel DataFrame"""
        eligibility_data = []
        try:
            # Look for eligibility section
            for idx, row in df.iterrows():
                row_str = ' '.join([str(cell) for cell in row if pd.notna(cell)]).upper()
                
                if any(marker in row_str for marker in ['CIVIL SERVICE ELIGIBILITY', 'ELIGIBILITY', 'CAREER SERVICE']):
                    # Extract eligibility data
                    for i in range(idx + 1, min(idx + 10, len(df))):
                        elig_row = df.iloc[i]
                        elig_values = [str(cell) for cell in elig_row if pd.notna(cell) and str(cell).strip() != '']
                        
                        if len(elig_values) >= 2:
                            eligibility_data.append({
                                'eligibility': elig_values[0] if len(elig_values) > 0 else 'N/A',
                                'rating': elig_values[1] if len(elig_values) > 1 else 'N/A',
                                'date_exam': elig_values[2] if len(elig_values) > 2 else 'N/A',
                                'place_exam': elig_values[3] if len(elig_values) > 3 else 'N/A'
                            })
                    break
        except Exception as e:
            logger.warning(f"Error extracting eligibility: {e}")
        
        return eligibility_data

    def _extract_voluntary_work_from_excel(self, df):
        """Extract voluntary work from Excel DataFrame"""
        voluntary_data = []
        try:
            # Look for voluntary work section
            for idx, row in df.iterrows():
                row_str = ' '.join([str(cell) for cell in row if pd.notna(cell)]).upper()
                
                if any(marker in row_str for marker in ['VOLUNTARY WORK', 'VOLUNTEER', 'COMMUNITY SERVICE']):
                    # Extract voluntary work data
                    for i in range(idx + 1, min(idx + 10, len(df))):
                        vol_row = df.iloc[i]
                        vol_values = [str(cell) for cell in vol_row if pd.notna(cell) and str(cell).strip() != '']
                        
                        if len(vol_values) >= 2:
                            hours = 0
                            try:
                                hours = float(vol_values[2]) if len(vol_values) > 2 else 0
                            except:
                                hours = 0
                            
                            voluntary_data.append({
                                'organization': vol_values[0] if len(vol_values) > 0 else 'N/A',
                                'position': vol_values[1] if len(vol_values) > 1 else 'N/A',
                                'hours': hours
                            })
                    break
        except Exception as e:
            logger.warning(f"Error extracting voluntary work: {e}")
        
        return voluntary_data

    def process_pds_candidate(self, content):
        """Process PDS candidate and extract relevant information."""
        try:
            # Extract basic information
            candidate_info = self.extract_basic_info(content)
            
            # Extract education with structured format
            education = self.extract_education_pds(content)
            
            # Extract work experience
            experience = self.extract_experience_pds(content)
            
            # Extract skills and competencies
            skills = self.extract_skills_pds(content)
            
            return {
                'basic_info': candidate_info,
                'education': education,
                'experience': experience,
                'skills': skills,
                'raw_content': content
            }
            
        except Exception as e:
            self.logger.error(f"Error processing PDS candidate: {str(e)}")
            return {}
    
    def convert_pds_to_candidate_format(self, pds_data):
        """Convert PDS data to standardized candidate format."""
        try:
            basic_info = pds_data.get('basic_info', {})
            education = pds_data.get('education', [])
            experience = pds_data.get('experience', [])
            skills = pds_data.get('skills', [])
            
            # Format education for display
            education_text = ""
            if education:
                for edu in education:
                    if isinstance(edu, dict):
                        degree = edu.get('degree', '')
                        school = edu.get('school', '')
                        year = edu.get('year', '')
                        education_text += f"{degree} from {school} ({year})\n"
                    else:
                        education_text += f"{edu}\n"
            
            # Format experience for display
            experience_text = ""
            if experience:
                for exp in experience:
                    if isinstance(exp, dict):
                        position = exp.get('position', '')
                        company = exp.get('company', '')
                        duration = exp.get('duration', '')
                        experience_text += f"{position} at {company} ({duration})\n"
                    else:
                        experience_text += f"{exp}\n"
            
            # Format skills
            skills_text = ", ".join(skills) if isinstance(skills, list) else str(skills)
            
            return {
                'name': basic_info.get('name', 'N/A'),
                'email': basic_info.get('email', 'N/A'),
                'phone': basic_info.get('phone', 'N/A'),
                'education': education_text.strip(),
                'experience': experience_text.strip(),
                'skills': skills_text,
                'additional_info': basic_info
            }
            
        except Exception as e:
            self.logger.error(f"Error converting PDS to candidate format: {str(e)}")
            return {}
    
    def preprocess_text(self, text):
        """Preprocess text for analysis."""
        if not text:
            return ""
        
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\-\+\#\.\@\%]', ' ', text)
        
        return text

    def extract_pdf_text(self, file_path):
        """Extract text from PDF files."""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_pds_information(self, text: str, filename: str = "") -> Dict[str, Any]:
        """Extract comprehensive information from a Personal Data Sheet."""
        try:
            # Check if this is a Philippine Civil Service Commission format
            is_csc_format = self._is_csc_format(text)
            
            # Basic information
            basic_info = self.extract_basic_info(text)
            
            # Enhanced PDS-specific extraction
            pds_data = {
                'filename': filename,
                'is_csc_format': is_csc_format,
                'basic_info': basic_info,
                'personal_information': self.extract_personal_information_pds(text),
                'family_background': self.extract_family_background(text),
                'education': self.extract_education_detailed(text),
                'experience': self.extract_experience_detailed(text),
                'skills': self.extract_skills_categorized(text),
                'certifications': self.extract_certifications(text),
                'training': self.extract_training_seminars(text),
                'awards': self.extract_awards_recognition(text),
                'eligibility': self.extract_civil_service_eligibility(text),
                'languages': self.extract_language_proficiency(text),
                'licenses': self.extract_licenses(text),
                'volunteer_work': self.extract_volunteer_work(text),
                'personal_references': self.extract_references(text),
                'government_id': self.extract_government_ids(text),
                'other_information': self.extract_other_information(text)
            }
            
            # Apply CSC-specific parsing if detected
            if is_csc_format:
                pds_data = self._enhance_csc_parsing(pds_data, text)
            
            return pds_data
            
        except Exception as e:
            self.logger.error(f"Error extracting PDS information: {str(e)}")
            return {'error': str(e)}
    
    def extract_skills_categorized(self, text: str) -> Dict[str, List[str]]:
        """Extract skills and categorize them for PDS analysis."""
        # Use the parent class's extract_skills method
        all_skills = self.extract_skills(text)
        
        # Categorize skills
        categorized = {
            'technical': [],
            'soft': [],
            'language': [],
            'certifications': []
        }
        
        # Technical skills keywords
        technical_keywords = ['python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 'vue', 
                            'node.js', 'django', 'flask', 'spring', 'docker', 'kubernetes', 'aws', 'azure', 
                            'git', 'linux', 'windows', 'mysql', 'postgresql', 'mongodb', 'excel', 'powerpoint',
                            'photoshop', 'autocad', 'microsoft office', 'data analysis', 'machine learning']
        
        # Soft skills keywords
        soft_keywords = ['leadership', 'communication', 'teamwork', 'problem solving', 'time management',
                        'project management', 'analytical', 'creative', 'adaptable', 'organized']
        
        # Language keywords
        language_keywords = ['english', 'filipino', 'tagalog', 'spanish', 'chinese', 'japanese', 'korean']
        
        for skill in all_skills:
            skill_lower = skill.lower()
            
            # Check if it's a technical skill
            if any(tech in skill_lower for tech in technical_keywords):
                categorized['technical'].append(skill)
            # Check if it's a soft skill
            elif any(soft in skill_lower for soft in soft_keywords):
                categorized['soft'].append(skill)
            # Check if it's a language skill
            elif any(lang in skill_lower for lang in language_keywords):
                categorized['language'].append(skill)
            else:
                # Default to technical if uncertain
                categorized['technical'].append(skill)
        
        return categorized
    
    def _is_csc_format(self, text: str) -> bool:
        """Detect if the PDS follows Philippine Civil Service Commission format."""
        csc_indicators = [
            'CS Form No. 212',
            'Personal Data Sheet',
            'Civil Service Commission',
            'Republic of the Philippines',
            'PERSONAL INFORMATION',
            'FAMILY BACKGROUND',
            'EDUCATIONAL BACKGROUND',
            'CIVIL SERVICE ELIGIBILITY',
            'WORK EXPERIENCE',
            'VOLUNTARY WORK',
            'LEARNING AND DEVELOPMENT',
            'OTHER INFORMATION'
        ]
        
        matches = sum(1 for indicator in csc_indicators if indicator.lower() in text.lower())
        return matches >= 3  # If at least 3 indicators are found
    
    def extract_personal_information_pds(self, text: str) -> Dict[str, Any]:
        """Extract personal information section specific to PDS format."""
        personal_info = {}
        
        # Pattern for Philippine PDS personal information
        patterns = {
            'surname': r'(?:SURNAME|Last\s+Name)\s*:?\s*([A-Za-z\s]+)',
            'first_name': r'(?:FIRST\s+NAME|Given\s+Name)\s*:?\s*([A-Za-z\s]+)',
            'middle_name': r'(?:MIDDLE\s+NAME|Middle\s+Initial)\s*:?\s*([A-Za-z\s\.]+)',
            'name_extension': r'(?:NAME\s+EXTENSION|Extension)\s*:?\s*([A-Za-z\.]+)',
            'date_of_birth': r'(?:DATE\s+OF\s+BIRTH|Birth\s+Date)\s*:?\s*([0-9\/\-]+)',
            'place_of_birth': r'(?:PLACE\s+OF\s+BIRTH|Birth\s+Place)\s*:?\s*([^,\n]+)',
            'sex': r'(?:SEX|Gender)\s*:?\s*(Male|Female|M|F)',
            'civil_status': r'(?:CIVIL\s+STATUS|Marital\s+Status)\s*:?\s*(Single|Married|Widowed|Separated|Divorced)',
            'height': r'(?:HEIGHT|Height)\s*:?\s*([0-9\.]+\s*(?:m|cm|ft|in)?)',
            'weight': r'(?:WEIGHT|Weight)\s*:?\s*([0-9\.]+\s*(?:kg|lbs|pounds)?)',
            'blood_type': r'(?:BLOOD\s+TYPE|Blood\s+Group)\s*:?\s*([A-Z]+[+-]?)',
            'gsis_id': r'(?:GSIS\s+ID)\s*(?:NO\.?)?\s*:?\s*([0-9\-]+)',
            'pag_ibig_id': r'(?:PAG-IBIG\s+ID|HDMF)\s*(?:NO\.?)?\s*:?\s*([0-9\-]+)',
            'philhealth_id': r'(?:PHILHEALTH)\s*(?:NO\.?)?\s*:?\s*([0-9\-]+)',
            'sss_id': r'(?:SSS)\s*(?:NO\.?)?\s*:?\s*([0-9\-]+)',
            'tin_id': r'(?:TIN)\s*(?:NO\.?)?\s*:?\s*([0-9\-]+)',
            'agency_employee_no': r'(?:AGENCY\s+EMPLOYEE\s+NO|Employee\s+No)\s*\.?\s*:?\s*([A-Z0-9\-]+)',
            'citizenship': r'(?:CITIZENSHIP|Nationality)\s*:?\s*([A-Za-z\s]+)',
            'residential_address': r'(?:RESIDENTIAL\s+ADDRESS|Home\s+Address)\s*:?\s*([^,\n]+(?:,\s*[^,\n]+)*)',
            'permanent_address': r'(?:PERMANENT\s+ADDRESS|Permanent\s+Add)\s*:?\s*([^,\n]+(?:,\s*[^,\n]+)*)',
            'zip_code': r'(?:ZIP\s+CODE|Postal\s+Code)\s*:?\s*([0-9]+)',
            'telephone_no': r'(?:TELEPHONE\s+NO|Phone|Tel)\s*\.?\s*:?\s*([0-9\-\+\(\)\s]+)',
            'mobile_no': r'(?:MOBILE\s+NO|Cell\s+Phone|Mobile)\s*\.?\s*:?\s*([0-9\-\+\(\)\s]+)',
            'email_address': r'(?:E-MAIL\s+ADDRESS|Email)\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                personal_info[field] = match.group(1).strip()
        
        return personal_info
    
    def extract_family_background(self, text: str) -> Dict[str, Any]:
        """Extract family background section from PDS."""
        family_info = {}
        
        # Extract spouse information
        spouse_patterns = {
            'spouse_surname': r'(?:SPOUSE\s+SURNAME|Spouse.*Last\s+Name)\s*:?\s*([A-Za-z\s]+)',
            'spouse_first_name': r'(?:SPOUSE\s+FIRST\s+NAME|Spouse.*First\s+Name)\s*:?\s*([A-Za-z\s]+)',
            'spouse_middle_name': r'(?:SPOUSE\s+MIDDLE\s+NAME|Spouse.*Middle\s+Name)\s*:?\s*([A-Za-z\s\.]+)',
            'spouse_occupation': r'(?:SPOUSE.*OCCUPATION|Spouse.*Work)\s*:?\s*([^,\n]+)',
            'spouse_employer': r'(?:SPOUSE.*EMPLOYER|Spouse.*Company)\s*:?\s*([^,\n]+)',
            'spouse_business_address': r'(?:SPOUSE.*BUSINESS\s+ADDRESS|Spouse.*Work\s+Address)\s*:?\s*([^,\n]+)',
            'spouse_telephone': r'(?:SPOUSE.*TELEPHONE|Spouse.*Phone)\s*:?\s*([0-9\-\+\(\)\s]+)'
        }
        
        spouse_info = {}
        for field, pattern in spouse_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                spouse_info[field] = match.group(1).strip()
        
        if spouse_info:
            family_info['spouse'] = spouse_info
        
        # Extract children information
        children = self._extract_children_info(text)
        if children:
            family_info['children'] = children
        
        # Extract father information
        father_patterns = {
            'father_surname': r'(?:FATHER.*SURNAME|Father.*Last\s+Name)\s*:?\s*([A-Za-z\s]+)',
            'father_first_name': r'(?:FATHER.*FIRST\s+NAME|Father.*First\s+Name)\s*:?\s*([A-Za-z\s]+)',
            'father_middle_name': r'(?:FATHER.*MIDDLE\s+NAME|Father.*Middle\s+Name)\s*:?\s*([A-Za-z\s\.]+)'
        }
        
        father_info = {}
        for field, pattern in father_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                father_info[field] = match.group(1).strip()
        
        if father_info:
            family_info['father'] = father_info
        
        # Extract mother information
        mother_patterns = {
            'mother_maiden_name': r'(?:MOTHER.*MAIDEN\s+NAME|Mother.*Maiden)\s*:?\s*([A-Za-z\s]+)',
            'mother_surname': r'(?:MOTHER.*SURNAME|Mother.*Last\s+Name)\s*:?\s*([A-Za-z\s]+)',
            'mother_first_name': r'(?:MOTHER.*FIRST\s+NAME|Mother.*First\s+Name)\s*:?\s*([A-Za-z\s]+)',
            'mother_middle_name': r'(?:MOTHER.*MIDDLE\s+NAME|Mother.*Middle\s+Name)\s*:?\s*([A-Za-z\s\.]+)'
        }
        
        mother_info = {}
        for field, pattern in mother_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                mother_info[field] = match.group(1).strip()
        
        if mother_info:
            family_info['mother'] = mother_info
        
        return family_info
    
    def _extract_children_info(self, text: str) -> List[Dict[str, Any]]:
        """Extract children information from family background section."""
        children = []
        
        # Look for children section
        children_section = re.search(r'(?:CHILDREN|CHILD|OFFSPRING).*?(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        
        if children_section:
            children_text = children_section.group(0)
            
            # Pattern for child information: Name and Date of Birth
            child_patterns = [
                r'([A-Za-z\s,]+)\s*(?:born|b\.?)\s*([0-9\/\-]+)',
                r'([A-Za-z\s,]+),?\s*([0-9\/\-]+)',
                r'Name:\s*([A-Za-z\s,]+).*?(?:Birth|DOB|Born):\s*([0-9\/\-]+)'
            ]
            
            for pattern in child_patterns:
                matches = re.findall(pattern, children_text, re.IGNORECASE)
                for match in matches:
                    if len(match) >= 2:
                        children.append({
                            'name': match[0].strip(),
                            'date_of_birth': match[1].strip()
                        })
        
        return children
    
    def extract_other_information(self, text: str) -> Dict[str, Any]:
        """Extract other information section from PDS."""
        other_info = {}
        
        # Special skills and hobbies
        skills_match = re.search(r'(?:SPECIAL\s+SKILLS|HOBBIES|OTHER\s+SKILLS)\s*:?\s*([^,\n]+(?:\n[^,\n]*)*)', text, re.IGNORECASE)
        if skills_match:
            other_info['special_skills'] = skills_match.group(1).strip()
        
        # Non-academic distinctions/recognition
        recognition_match = re.search(r'(?:NON-ACADEMIC\s+DISTINCTIONS|RECOGNITION|HONORS)\s*:?\s*([^,\n]+(?:\n[^,\n]*)*)', text, re.IGNORECASE)
        if recognition_match:
            other_info['recognition'] = recognition_match.group(1).strip()
        
        # Membership in association/organization
        membership_match = re.search(r'(?:MEMBERSHIP\s+IN\s+ASSOCIATION|ORGANIZATION|PROFESSIONAL\s+MEMBERSHIP)\s*:?\s*([^,\n]+(?:\n[^,\n]*)*)', text, re.IGNORECASE)
        if membership_match:
            other_info['memberships'] = membership_match.group(1).strip()
        
        # Questions/declarations (common in CSC forms)
        declarations = self._extract_declarations(text)
        if declarations:
            other_info['declarations'] = declarations
        
        return other_info
    
    def _extract_declarations(self, text: str) -> Dict[str, str]:
        """Extract declarations/questions section from CSC PDS."""
        declarations = {}
        
        # Common PDS declaration questions
        declaration_patterns = {
            'related_to_government_official': r'(?:related.*(?:third|3rd)\s+degree.*government\s+official)\s*:?\s*(Yes|No)',
            'administrative_offense': r'(?:administrative\s+offense.*guilty)\s*:?\s*(Yes|No)',
            'criminal_charge': r'(?:criminal\s+charge.*court)\s*:?\s*(Yes|No)',
            'conviction': r'(?:convicted.*crime)\s*:?\s*(Yes|No)',
            'separated_from_service': r'(?:separated.*service.*government)\s*:?\s*(Yes|No)',
            'election_candidate': r'(?:candidate.*election)\s*:?\s*(Yes|No)',
            'resigned_campaign': r'(?:resigned.*campaign)\s*:?\s*(Yes|No)',
            'immigrant_status': r'(?:immigrant.*dual\s+citizenship)\s*:?\s*(Yes|No)',
            'indigenous_group': r'(?:indigenous.*group)\s*:?\s*(Yes|No)',
            'pwd': r'(?:person.*disability|PWD)\s*:?\s*(Yes|No)',
            'solo_parent': r'(?:solo\s+parent)\s*:?\s*(Yes|No)'
        }
        
        for field, pattern in declaration_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                declarations[field] = match.group(1).strip()
        
        return declarations
    
    def _enhance_csc_parsing(self, pds_data: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Enhance parsing for CSC format specific elements."""
        # Add CSC-specific enhancements
        
        # Enhanced educational background parsing for CSC format
        if 'education' in pds_data:
            pds_data['education'] = self._enhance_csc_education_parsing(text)
        
        # Enhanced work experience parsing for CSC format
        if 'experience' in pds_data:
            pds_data['experience'] = self._enhance_csc_experience_parsing(text)
        
        # Parse Learning and Development section
        pds_data['learning_development'] = self._extract_learning_development(text)
        
        return pds_data
    
    def _enhance_csc_education_parsing(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced education parsing for CSC format."""
        education_list = []
        
        # CSC format typically has structured education sections
        education_section = re.search(r'(?:EDUCATIONAL\s+BACKGROUND|EDUCATION).*?(?:CIVIL\s+SERVICE\s+ELIGIBILITY|WORK\s+EXPERIENCE|\Z)', 
                                    text, re.IGNORECASE | re.DOTALL)
        
        if education_section:
            edu_text = education_section.group(0)
            
            # Parse different education levels
            levels = ['ELEMENTARY', 'SECONDARY', 'VOCATIONAL', 'COLLEGE', 'GRADUATE STUDIES']
            
            for level in levels:
                level_match = re.search(rf'{level}.*?(?:{"|".join([l for l in levels if l != level])}|\Z)', 
                                      edu_text, re.IGNORECASE | re.DOTALL)
                if level_match:
                    level_text = level_match.group(0)
                    
                    # Extract school name, degree, year
                    school_match = re.search(r'(?:NAME\s+OF\s+SCHOOL|SCHOOL)\s*:?\s*([^,\n]+)', level_text, re.IGNORECASE)
                    degree_match = re.search(r'(?:BASIC\s+EDUCATION|DEGREE|COURSE)\s*:?\s*([^,\n]+)', level_text, re.IGNORECASE)
                    year_match = re.search(r'(?:YEAR\s+GRADUATED|GRADUATED)\s*:?\s*(\d{4})', level_text, re.IGNORECASE)
                    highest_match = re.search(r'(?:HIGHEST\s+LEVEL|LEVEL)\s*:?\s*([^,\n]+)', level_text, re.IGNORECASE)
                    
                    if school_match:
                        education_entry = {
                            'level': level.lower().replace(' ', '_'),
                            'institution': school_match.group(1).strip(),
                            'degree': degree_match.group(1).strip() if degree_match else '',
                            'year_graduated': year_match.group(1) if year_match else None,
                            'highest_level': highest_match.group(1).strip() if highest_match else ''
                        }
                        education_list.append(education_entry)
        
        return education_list
    
    def _enhance_csc_experience_parsing(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced work experience parsing for CSC format."""
        experience_list = []
        
        # CSC format work experience section
        work_section = re.search(r'(?:WORK\s+EXPERIENCE|EMPLOYMENT\s+RECORD).*?(?:VOLUNTARY\s+WORK|LEARNING\s+AND\s+DEVELOPMENT|\Z)', 
                               text, re.IGNORECASE | re.DOTALL)
        
        if work_section:
            work_text = work_section.group(0)
            
            # CSC format typically has tabular data
            # Look for patterns like dates, position, department, salary
            experience_patterns = [
                r'(\d{1,2}\/\d{1,2}\/\d{4})\s*(?:to|-)?\s*(\d{1,2}\/\d{1,2}\/\d{4}|present)\s*([^,\n]+)\s*([^,\n]+)\s*([^,\n]*)',
                r'FROM:\s*(\d{1,2}\/\d{1,2}\/\d{4})\s*TO:\s*(\d{1,2}\/\d{1,2}\/\d{4}|present)\s*POSITION:\s*([^,\n]+)\s*DEPARTMENT:\s*([^,\n]+)'
            ]
            
            for pattern in experience_patterns:
                matches = re.findall(pattern, work_text, re.IGNORECASE)
                for match in matches:
                    if len(match) >= 4:
                        start_date = match[0]
                        end_date = match[1]
                        position = match[2].strip()
                        department = match[3].strip()
                        
                        # Convert dates to years for calculation
                        try:
                            start_year = int(start_date.split('/')[-1]) if '/' in start_date else None
                            end_year = datetime.now().year if end_date.lower() in ['present', 'current'] else (int(end_date.split('/')[-1]) if '/' in end_date else None)
                        except:
                            start_year = end_year = None
                        
                        experience_entry = {
                            'start_date': start_date,
                            'end_date': end_date,
                            'start_year': start_year,
                            'end_year': end_year,
                            'position': position,
                            'department_agency': department,
                            'monthly_salary': match[4].strip() if len(match) > 4 else '',
                            'duration_years': (end_year - start_year) if start_year and end_year else None
                        }
                        experience_list.append(experience_entry)
        
        return experience_list
    
    def _extract_learning_development(self, text: str) -> List[Dict[str, Any]]:
        """Extract Learning and Development interventions/training programs."""
        training_list = []
        
        # Look for L&D section
        ld_section = re.search(r'(?:LEARNING\s+AND\s+DEVELOPMENT|L&D|TRAINING\s+PROGRAMS?).*?(?:OTHER\s+INFORMATION|REFERENCES|\Z)', 
                             text, re.IGNORECASE | re.DOTALL)
        
        if ld_section:
            ld_text = ld_section.group(0)
            
            # Parse training information
            training_patterns = [
                r'(?:TITLE\s+OF\s+LEARNING|TRAINING\s+TITLE)\s*:?\s*([^,\n]+).*?(?:INCLUSIVE\s+DATES|DATES?)\s*:?\s*([^,\n]+).*?(?:NUMBER\s+OF\s+HOURS|HOURS?)\s*:?\s*([0-9]+)',
                r'([^,\n]+)\s*-\s*([^,\n]+)\s*\((\d+)\s*hours?\)',
                r'([^,\n]+),\s*([^,\n]+),\s*(\d+)\s*(?:hours?|hrs?)'
            ]
            
            for pattern in training_patterns:
                matches = re.findall(pattern, ld_text, re.IGNORECASE)
                for match in matches:
                    if len(match) >= 3:
                        training_entry = {
                            'title': match[0].strip(),
                            'dates': match[1].strip(),
                            'hours': int(match[2]) if match[2].isdigit() else 0,
                            'type': 'learning_development'
                        }
                        training_list.append(training_entry)
        
        return training_list
    
    def extract_education_detailed(self, text: str) -> List[Dict[str, Any]]:
        """Extract detailed education information specific to PDS format."""
        education_list = []
        
        # Enhanced patterns for PDS education format
        education_patterns = [
            # Standard format: Degree, Institution, Year, GPA/Honors
            r'(?:Bachelor|Master|PhD|Doctorate|BS|BA|MS|MA|BSc|MSc|BSIT|BSCS|MIT|MBA)\s+(?:of|in|degree in)?\s*([^,\n]+),?\s*([^,\n]+),?\s*(\d{4})\s*(?:GPA[:\s]*([0-9.]+)|([^,\n]*honors?))?',
            
            # Alternative format with dates
            r'([^,\n]+)\s*-\s*([^,\n]+)\s*\((\d{4})\s*-?\s*(\d{4})?\)',
            
            # Simple format
            r'Education[:\s]*([^,\n]+),?\s*([^,\n]+),?\s*(\d{4})'
        ]
        
        for pattern in education_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if len(match) >= 3:
                    education_entry = {
                        'degree': match[0].strip(),
                        'institution': match[1].strip(),
                        'year': match[2],
                        'gpa': match[3] if len(match) > 3 and match[3] else None,
                        'honors': match[4] if len(match) > 4 and match[4] else None
                    }
                    education_list.append(education_entry)
        
        return education_list
    
    def extract_experience_detailed(self, text: str) -> List[Dict[str, Any]]:
        """Extract detailed work experience information from PDS."""
        experience_list = []
        
        # Enhanced patterns for work experience
        experience_patterns = [
            # Standard format: Position, Company, Start-End dates
            r'(?:Position|Job Title|Work)\s*:?\s*([^,\n]+),?\s*([^,\n]+),?\s*(\d{4})\s*-\s*(\d{4}|present|current)',
            
            # Alternative format
            r'([^,\n]+)\s*-\s*([^,\n]+)\s*\((\d{4})\s*-\s*(\d{4}|present|current)\)',
            
            # Simple format with company
            r'([A-Za-z\s]+),\s*([A-Za-z\s&.,]+),?\s*(\d{4})\s*-?\s*(\d{4}|present|current)?'
        ]
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if len(match) >= 3:
                    start_year = int(match[2]) if match[2].isdigit() else None
                    end_year = datetime.now().year if match[3].lower() in ['present', 'current'] else (int(match[3]) if match[3].isdigit() else None)
                    
                    experience_entry = {
                        'position': match[0].strip(),
                        'company': match[1].strip(),
                        'start_year': start_year,
                        'end_year': end_year,
                        'duration_years': (end_year - start_year) if start_year and end_year else None,
                        'description': ''  # Could be enhanced to extract job descriptions
                    }
                    experience_list.append(experience_entry)
        
        return experience_list
    
    def extract_certifications(self, text: str) -> List[Dict[str, Any]]:
        """Extract certifications and professional licenses."""
        certifications = []
        
        # Patterns for certifications
        cert_patterns = [
            r'(?:Certification|Certificate|Certified|License)\s*:?\s*([^,\n]+)(?:,\s*(\d{4}|\w+\s+\d{4}))?',
            r'([A-Z]{2,})\s+(?:Certification|Certificate|Certified)(?:\s*-\s*(\d{4}))?',
            r'Professional\s+License\s*:?\s*([^,\n]+)'
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cert_name = match[0] if isinstance(match, tuple) else match
                issue_date = match[1] if isinstance(match, tuple) and len(match) > 1 else None
                
                certifications.append({
                    'name': cert_name.strip(),
                    'issue_date': issue_date,
                    'type': 'certification'
                })
        
        return certifications
    
    def extract_civil_service_eligibility(self, text: str) -> List[Dict[str, Any]]:
        """Extract civil service eligibility information."""
        eligibility_list = []
        
        eligibility_patterns = [
            r'(?:Civil\s+Service|CSE|Career\s+Service)\s+(?:Eligibility|Examination|Exam)\s*:?\s*([^,\n]+)(?:,\s*(\d{4}))?',
            r'Eligibility\s*:?\s*([^,\n]*(?:Professional|Sub-professional|Career\s+Service)[^,\n]*)',
            r'(?:Professional|Sub-professional)\s+(?:Board|Examination|Exam)\s*:?\s*([^,\n]+)'
        ]
        
        for pattern in eligibility_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                eligibility_name = match[0] if isinstance(match, tuple) else match
                exam_date = match[1] if isinstance(match, tuple) and len(match) > 1 else None
                
                eligibility_list.append({
                    'type': eligibility_name.strip(),
                    'date_taken': exam_date,
                    'status': 'passed'  # Assuming passed if listed
                })
        
        return eligibility_list
    
    def extract_training_seminars(self, text: str) -> List[Dict[str, Any]]:
        """Extract training programs and seminars attended."""
        training_list = []
        
        training_patterns = [
            r'(?:Training|Seminar|Workshop|Course)\s*:?\s*([^,\n]+)(?:,\s*([^,\n]+))(?:,\s*(\d{4}|\w+\s+\d{4}))?',
            r'([^,\n]+)\s+(?:Training|Seminar|Workshop)\s*(?:-\s*([^,\n]+))?(?:,\s*(\d{4}))?'
        ]
        
        for pattern in training_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                training_name = match[0].strip()
                provider = match[1].strip() if len(match) > 1 and match[1] else None
                date = match[2] if len(match) > 2 and match[2] else None
                
                training_list.append({
                    'name': training_name,
                    'provider': provider,
                    'date': date,
                    'type': 'training'
                })
        
        return training_list
    
    def extract_awards_recognition(self, text: str) -> List[Dict[str, Any]]:
        """Extract awards and recognition."""
        awards_list = []
        
        award_patterns = [
            r'(?:Award|Recognition|Honor|Achievement)\s*:?\s*([^,\n]+)(?:,\s*(\d{4}|\w+\s+\d{4}))?',
            r'([^,\n]*(?:Award|Prize|Medal|Honor)[^,\n]*)(?:,\s*(\d{4}))?'
        ]
        
        for pattern in award_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                award_name = match[0] if isinstance(match, tuple) else match
                year = match[1] if isinstance(match, tuple) and len(match) > 1 else None
                
                awards_list.append({
                    'name': award_name.strip(),
                    'year': year,
                    'type': 'award'
                })
        
        return awards_list
    
    def extract_language_proficiency(self, text: str) -> List[Dict[str, Any]]:
        """Extract language proficiency information."""
        languages = []
        
        # Common languages and proficiency levels
        common_languages = ['English', 'Filipino', 'Tagalog', 'Spanish', 'Chinese', 'Japanese', 'Korean', 'French', 'German']
        proficiency_levels = ['Native', 'Fluent', 'Proficient', 'Intermediate', 'Basic', 'Conversational']
        
        language_patterns = [
            r'(?:Language|Languages)\s*:?\s*([^,\n]+)',
            r'([A-Za-z]+)\s*[-:]\s*(Native|Fluent|Proficient|Intermediate|Basic|Conversational)',
            r'(English|Filipino|Tagalog|Spanish|Chinese|Japanese|Korean|French|German)\s*[-:]?\s*(Native|Fluent|Proficient|Intermediate|Basic|Conversational)?'
        ]
        
        for pattern in language_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    language = match[0].strip()
                    proficiency = match[1].strip() if len(match) > 1 and match[1] else 'Proficient'
                else:
                    language = match.strip()
                    proficiency = 'Proficient'
                
                if language in common_languages:
                    languages.append({
                        'language': language,
                        'proficiency': proficiency
                    })
        
        return languages
    
    def extract_licenses(self, text: str) -> List[Dict[str, Any]]:
        """Extract professional licenses."""
        licenses = []
        
        license_patterns = [
            r'(?:License|Licensed)\s*:?\s*([^,\n]+)(?:,?\s*License\s*No\.?\s*([A-Z0-9\-]+))?(?:,?\s*(\d{4}))?',
            r'([A-Z]{2,})\s+License(?:\s*No\.?\s*([A-Z0-9\-]+))?(?:,?\s*(\d{4}))?'
        ]
        
        for pattern in license_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                license_type = match[0].strip()
                license_number = match[1] if len(match) > 1 and match[1] else None
                issue_year = match[2] if len(match) > 2 and match[2] else None
                
                licenses.append({
                    'type': license_type,
                    'number': license_number,
                    'issue_year': issue_year
                })
        
        return licenses
    
    def extract_volunteer_work(self, text: str) -> List[Dict[str, Any]]:
        """Extract volunteer work and community service."""
        volunteer_work = []
        
        volunteer_patterns = [
            r'(?:Volunteer|Community\s+Service|Civic\s+Activities)\s*:?\s*([^,\n]+)(?:,\s*([^,\n]+))?(?:,\s*(\d{4}))?',
            r'([^,\n]+)\s*-\s*Volunteer(?:\s*at\s*([^,\n]+))?(?:,\s*(\d{4}))?'
        ]
        
        for pattern in volunteer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                activity = match[0].strip()
                organization = match[1].strip() if len(match) > 1 and match[1] else None
                year = match[2] if len(match) > 2 and match[2] else None
                
                volunteer_work.append({
                    'activity': activity,
                    'organization': organization,
                    'year': year
                })
        
        return volunteer_work
    
    def extract_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract personal references."""
        references = []
        
        # Look for reference section
        reference_section = re.search(r'(?:References?|Character\s+References?)\s*:?\s*(.*?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        
        if reference_section:
            ref_text = reference_section.group(1)
            
            # Pattern for name, position, contact
            ref_patterns = [
                r'([A-Za-z\s\.]+),?\s*([^,\n]+),?\s*([0-9\-\+\(\)\s]+|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                r'([A-Za-z\s\.]+)\s*-\s*([^,\n]+)'
            ]
            
            for pattern in ref_patterns:
                matches = re.findall(pattern, ref_text, re.IGNORECASE)
                for match in matches:
                    name = match[0].strip()
                    position = match[1].strip() if len(match) > 1 else None
                    contact = match[2].strip() if len(match) > 2 else None
                    
                    references.append({
                        'name': name,
                        'position': position,
                        'contact': contact
                    })
        
        return references
    
    def extract_government_ids(self, text: str) -> Dict[str, str]:
        """Extract government ID numbers."""
        gov_ids = {}
        
        id_patterns = {
            'sss': r'(?:SSS|Social\s+Security)\s*(?:No\.?|Number)\s*:?\s*([0-9\-]+)',
            'tin': r'(?:TIN|Tax\s+Identification)\s*(?:No\.?|Number)\s*:?\s*([0-9\-]+)',
            'philhealth': r'(?:PhilHealth|Phil\s*Health)\s*(?:No\.?|Number)\s*:?\s*([0-9\-]+)',
            'pagibig': r'(?:Pag-IBIG|HDMF)\s*(?:No\.?|Number)\s*:?\s*([0-9\-]+)',
            'passport': r'(?:Passport)\s*(?:No\.?|Number)\s*:?\s*([A-Z0-9]+)',
            'drivers_license': r'(?:Driver\'?s?\s+License|DL)\s*(?:No\.?|Number)\s*:?\s*([A-Z0-9\-]+)'
        }
        
        for id_type, pattern in id_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gov_ids[id_type] = match.group(1).strip()
        
        return gov_ids
    
    def score_pds_against_job(self, pds_data: Dict[str, Any], job_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Score a Personal Data Sheet against job requirements using configurable criteria."""
        try:
            scores = {}
            total_score = 0
            
            # Education scoring
            education_score = self._score_education(pds_data.get('education', []), job_requirements)
            scores['education'] = education_score
            total_score += education_score * self.pds_scoring_criteria['education']['weight']
            
            # Experience scoring
            experience_score = self._score_experience(pds_data.get('experience', []), job_requirements)
            scores['experience'] = experience_score
            total_score += experience_score * self.pds_scoring_criteria['experience']['weight']
            
            # Skills scoring
            skills_score = self._score_skills_pds(pds_data.get('skills', {}), job_requirements)
            scores['skills'] = skills_score
            total_score += skills_score * self.pds_scoring_criteria['skills']['weight']
            
            # Personal attributes scoring
            personal_score = self._score_personal_attributes(pds_data, job_requirements)
            scores['personal_attributes'] = personal_score
            total_score += personal_score * self.pds_scoring_criteria['personal_attributes']['weight']
            
            # Additional qualifications scoring
            additional_score = self._score_additional_qualifications(pds_data, job_requirements)
            scores['additional_qualifications'] = additional_score
            total_score += additional_score * self.pds_scoring_criteria['additional_qualifications']['weight']
            
            return {
                'total_score': round(total_score, 2),
                'category_scores': scores,
                'scoring_breakdown': self._generate_scoring_breakdown(scores, pds_data, job_requirements)
            }
            
        except Exception as e:
            self.logger.error(f"Error scoring PDS: {str(e)}")
            return {'total_score': 0, 'error': str(e)}
    
    def _score_education(self, education_list: List[Dict], job_requirements: Dict) -> float:
        """Score education based on relevance, level, and institution."""
        if not education_list:
            return 0
        
        required_education = job_requirements.get('education_level', '').lower()
        preferred_field = job_requirements.get('preferred_field', '').lower()
        
        max_score = 0
        for edu in education_list:
            score = 0
            degree = edu.get('degree', '').lower()
            
            # Education level scoring
            if 'phd' in degree or 'doctorate' in degree:
                level_score = 100
            elif 'master' in degree or 'ms' in degree or 'ma' in degree:
                level_score = 85
            elif 'bachelor' in degree or 'bs' in degree or 'ba' in degree:
                level_score = 70
            else:
                level_score = 50
            
            # Field relevance scoring
            relevance_score = 70  # Base score
            if preferred_field and preferred_field in degree:
                relevance_score = 100
            
            # Institution scoring (simplified)
            institution_score = 75  # Base score for any accredited institution
            
            # Grades scoring
            grades_score = 75  # Base score
            if edu.get('honors'):
                grades_score = 90
            if edu.get('gpa') and float(edu.get('gpa', 0)) >= 3.5:
                grades_score = max(grades_score, 85)
            
            # Weighted calculation
            weighted_score = (
                relevance_score * 0.4 +
                level_score * 0.3 +
                institution_score * 0.2 +
                grades_score * 0.1
            )
            
            max_score = max(max_score, weighted_score)
        
        return max_score
    
    def _score_experience(self, experience_list: List[Dict], job_requirements: Dict) -> float:
        """Score work experience based on relevance and duration."""
        if not experience_list:
            return 0
        
        required_years = job_requirements.get('experience_years', 0)
        relevant_keywords = job_requirements.get('relevant_experience', [])
        
        total_years = 0
        relevance_score = 0
        
        for exp in experience_list:
            # Calculate years of experience
            start_year = exp.get('start_year', 0)
            end_year = exp.get('end_year', datetime.now().year)
            if start_year:
                years = end_year - start_year
                total_years += years
            
            # Check relevance
            job_title = exp.get('position', '').lower()
            company = exp.get('company', '').lower()
            description = exp.get('description', '').lower()
            
            exp_text = f"{job_title} {company} {description}"
            keyword_matches = sum(1 for keyword in relevant_keywords if keyword.lower() in exp_text)
            
            if keyword_matches > 0:
                relevance_score = max(relevance_score, min(100, keyword_matches * 25))
        
        # Duration scoring
        duration_score = min(100, (total_years / max(required_years, 1)) * 100)
        
        # Responsibilities scoring (simplified)
        responsibilities_score = 75  # Base score
        
        return (
            relevance_score * 0.5 +
            duration_score * 0.3 +
            responsibilities_score * 0.2
        )
    
    def _score_skills_pds(self, skills_data: Dict, job_requirements: Dict) -> float:
        """Score skills with emphasis on technical match and certifications."""
        required_skills = job_requirements.get('required_skills', [])
        if not required_skills:
            return 75  # Base score if no specific requirements
        
        technical_skills = skills_data.get('technical', [])
        certifications = skills_data.get('certifications', [])
        
        # Technical skills matching
        matched_skills = 0
        for req_skill in required_skills:
            if any(req_skill.lower() in tech_skill.lower() for tech_skill in technical_skills):
                matched_skills += 1
        
        technical_score = (matched_skills / len(required_skills)) * 100 if required_skills else 0
        
        # Certifications scoring
        cert_score = min(100, len(certifications) * 20)  # 20 points per certification, max 100
        
        return technical_score * 0.6 + cert_score * 0.4
    
    def _score_personal_attributes(self, pds_data: Dict, job_requirements: Dict) -> float:
        """Score personal attributes like eligibility, awards, and training."""
        eligibility = pds_data.get('eligibility', [])
        awards = pds_data.get('awards', [])
        training = pds_data.get('training', [])
        
        # Eligibility scoring
        eligibility_score = min(100, len(eligibility) * 30)  # 30 points per eligibility
        
        # Awards scoring
        awards_score = min(100, len(awards) * 25)  # 25 points per award
        
        # Training scoring
        training_score = min(100, len(training) * 15)  # 15 points per training
        
        return (
            eligibility_score * 0.5 +
            awards_score * 0.3 +
            training_score * 0.2
        )
    
    def _score_additional_qualifications(self, pds_data: Dict, job_requirements: Dict) -> float:
        """Score additional qualifications like languages, licenses, and volunteer work."""
        languages = pds_data.get('languages', [])
        licenses = pds_data.get('licenses', [])
        volunteer_work = pds_data.get('volunteer_work', [])
        
        # Language scoring
        language_score = min(100, len(languages) * 25)  # 25 points per language
        
        # License scoring
        license_score = min(100, len(licenses) * 35)  # 35 points per license
        
        # Volunteer work scoring
        volunteer_score = min(100, len(volunteer_work) * 20)  # 20 points per volunteer activity
        
        return (
            language_score * 0.4 +
            license_score * 0.3 +
            volunteer_score * 0.3
        )
    
    def _generate_scoring_breakdown(self, scores: Dict, pds_data: Dict, job_requirements: Dict) -> Dict[str, Any]:
        """Generate detailed scoring breakdown for transparency."""
        return {
            'education': {
                'score': scores.get('education', 0),
                'weight': self.pds_scoring_criteria['education']['weight'],
                'details': f"Based on {len(pds_data.get('education', []))} education entries"
            },
            'experience': {
                'score': scores.get('experience', 0),
                'weight': self.pds_scoring_criteria['experience']['weight'],
                'details': f"Based on {len(pds_data.get('experience', []))} work experiences"
            },
            'skills': {
                'score': scores.get('skills', 0),
                'weight': self.pds_scoring_criteria['skills']['weight'],
                'details': f"Technical skills and certifications assessment"
            },
            'personal_attributes': {
                'score': scores.get('personal_attributes', 0),
                'weight': self.pds_scoring_criteria['personal_attributes']['weight'],
                'details': f"Eligibility, awards, and training evaluation"
            },
            'additional_qualifications': {
                'score': scores.get('additional_qualifications', 0),
                'weight': self.pds_scoring_criteria['additional_qualifications']['weight'],
                'details': f"Languages, licenses, and volunteer work"
            }
        }

    def process_pdf_file(self, file_path: str, filename: str, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process PDF file through complete PDS extraction and scoring pipeline
        Based on test_real_data_only.py comprehensive extraction process
        """
        try:
            # Step 1: Extract comprehensive PDS data (like test_real_data_only.py)
            print(f"ðŸ” Starting comprehensive PDS extraction for: {filename}")
            extracted_pds_data = self.extract_pds_data(file_path)
            
            if not extracted_pds_data:
                raise ValueError("No PDS data could be extracted")
            
            print(f"âœ… Raw PDS extraction successful! Sections found: {list(extracted_pds_data.keys())}")
            
            # Step 2: Convert to assessment format (copy from test_real_data_only.py)
            converted_data = self._convert_pds_to_comprehensive_format(extracted_pds_data, filename)
            
            # Step 3: Extract personal info manually (like test_real_data_only.py)
            try:
                personal_info = self._extract_personal_info_from_file(file_path, filename)
                converted_data['basic_info'].update(personal_info)
            except Exception as e:
                print(f"âš ï¸ Could not extract detailed personal info: {e}")
            
            # Step 4: Run comprehensive assessment using UniversityAssessmentEngine
            try:
                from database import DatabaseManager
                from assessment_engine import UniversityAssessmentEngine
                
                db_manager = DatabaseManager()
                assessment_engine = UniversityAssessmentEngine(db_manager)
                
                # Prepare job data for assessment
                job_for_assessment = {
                    'id': job['id'],
                    'position_title': job.get('title', ''),
                    'education_requirements': job.get('education_requirements', ''),
                    'experience_requirements': job.get('experience_requirements', ''),
                    'skills_requirements': job.get('skills_requirements', ''),
                    'preferred_qualifications': job.get('preferred_qualifications', ''),
                    'position_type_id': 1  # Default for LSPU jobs
                }
                
                print(f"ðŸŽ¯ Running comprehensive assessment for: {converted_data['basic_info']['name']}")
                
                # Run the real assessment engine
                assessment_result = assessment_engine.assess_candidate_for_lspu_job(
                    candidate_data=converted_data,
                    lspu_job=job_for_assessment,
                    position_type_id=job_for_assessment.get('position_type_id', 1)
                )
                
                print(f"âœ… Assessment complete! Score: {assessment_result.get('automated_score', 0):.2f}")
                
                # Convert assessment result to candidate format
                candidate_data = {
                    'name': converted_data['basic_info']['name'],
                    'email': converted_data['basic_info'].get('email', ''),
                    'phone': converted_data['basic_info'].get('phone', ''),
                    'address': converted_data['basic_info'].get('address', ''),
                    'resume_text': self._create_resume_text_from_pds(extracted_pds_data),
                    'job_id': job['id'],
                    'score': assessment_result.get('automated_score', 0),
                    'percentage_score': assessment_result.get('percentage_score', 0),
                    'processing_type': 'comprehensive_pds_extraction',
                    'original_filename': filename,
                    'assessment_results': assessment_result.get('assessment_results', {}),
                    'recommendation': assessment_result.get('recommendation', 'Unknown'),
                    'pds_data': extracted_pds_data,
                    'converted_data': converted_data
                }
                
                return candidate_data
                
            except Exception as assessment_error:
                print(f"âš ï¸ Assessment engine failed, using PDS scoring: {assessment_error}")
                
                # Fallback to direct PDS scoring if assessment engine fails
                job_requirements = {
                    'education_requirements': job.get('education_requirements', ''),
                    'experience_requirements': job.get('experience_requirements', ''),
                    'skills_requirements': job.get('skills_requirements', ''),
                    'preferred_qualifications': job.get('preferred_qualifications', ''),
                    'title': job.get('title', ''),
                    'category': job.get('category', '')
                }
                
                scoring_result = self.score_pds_against_job(extracted_pds_data, job_requirements)
                
                candidate_data = {
                    'name': converted_data['basic_info']['name'],
                    'email': converted_data['basic_info'].get('email', ''),
                    'phone': converted_data['basic_info'].get('phone', ''),
                    'address': converted_data['basic_info'].get('address', ''),
                    'resume_text': self._create_resume_text_from_pds(extracted_pds_data),
                    'job_id': job['id'],
                    'score': scoring_result.get('total_score', 0),
                    'percentage_score': scoring_result.get('percentage', 0),
                    'processing_type': 'pds_extraction_fallback',
                    'original_filename': filename,
                    'scoring_breakdown': scoring_result.get('breakdown', {}),
                    'rating': scoring_result.get('rating', 'Not Rated'),
                    'pds_data': extracted_pds_data,
                    'converted_data': converted_data
                }
                
                return candidate_data
            
        except Exception as e:
            print(f"âŒ Comprehensive PDS processing failed for {filename}: {e}")
            import traceback
            traceback.print_exc()
            
            # Final fallback to basic extraction
            try:
                import PyPDF2
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                
                return {
                    'name': filename.replace('.pdf', ''),
                    'email': '',
                    'phone': '',
                    'resume_text': text,
                    'job_id': job['id'],
                    'score': 15,  # Very low score for fallback
                    'percentage_score': 15,
                    'processing_type': 'basic_fallback',
                    'error': str(e)
                }
            except Exception as final_error:
                print(f"âŒ Even basic fallback failed for {filename}: {final_error}")
                return None

    def _convert_pds_to_comprehensive_format(self, extracted_data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """
        Convert extracted PDS data to comprehensive assessment format
        Based on convert_pds_to_assessment_format from test_real_data_only.py
        """
        converted_data = {
            'basic_info': {
                'name': 'Unknown Candidate',
                'email': 'candidate@example.com',
                'phone': 'N/A',
                'address': 'N/A',
                'citizenship': 'N/A',
                'civil_status': 'N/A',
                'birth_date': 'N/A',
                'birth_place': 'N/A'
            },
            'education': [],
            'experience': [],
            'experience_data': [],  # For assessment engine compatibility
            'training': [],
            'eligibility': [],
            'certifications': [],
            'awards': [],
            'volunteer_work': []
        }
        
        print(f"ðŸ”„ Converting PDS data to comprehensive format...")
        
        # Educational background
        if 'educational_background' in extracted_data:
            education = extracted_data['educational_background']
            if isinstance(education, list):
                for edu in education:
                    if edu and edu.get('level') and edu.get('level') not in ['N/a', '', 'nan']:
                        converted_data['education'].append({
                            'level': edu.get('level', 'N/A'),
                            'school': edu.get('school', 'N/A'),
                            'degree_course': edu.get('degree_course', 'N/A'),
                            'year_graduated': edu.get('year_graduated', 'N/A'),
                            'honors': edu.get('honors', 'N/A'),
                            'units_earned': edu.get('highest_level_units', 'N/A')
                        })
        
        # Work experience
        if 'work_experience' in extracted_data:
            experience = extracted_data['work_experience']
            if isinstance(experience, list):
                for exp in experience:
                    if exp and exp.get('position'):
                        experience_entry = {
                            'position': exp.get('position', 'N/A'),
                            'company': exp.get('company', 'N/A'),
                            'from_date': exp.get('date_from', 'N/A'),
                            'to_date': exp.get('date_to', 'N/A'),
                            'monthly_salary': exp.get('salary', 'N/A'),
                            'salary_grade': exp.get('grade', 'N/A'),
                            'govt_service': 'Y' if 'government' in str(exp.get('company', '')).lower() or 'civil service' in str(exp.get('company', '')).lower() or 'deped' in str(exp.get('company', '')).lower() else 'N'
                        }
                        # Add to both fields for compatibility
                        converted_data['experience'].append(experience_entry)
                        converted_data['experience_data'].append(experience_entry)
        
        # Training and seminars
        if 'learning_development' in extracted_data:
            training = extracted_data['learning_development']
            if isinstance(training, list):
                for train in training:
                    if train and train.get('title'):
                        hours = train.get('hours', 0)
                        try:
                            hours = float(hours) if hours else 0
                        except:
                            hours = 0
                        
                        converted_data['training'].append({
                            'title': train.get('title', 'N/A'),
                            'hours': hours,
                            'type': train.get('type', 'N/A'),
                            'provider': train.get('conductor', 'N/A')
                        })
        
        # Civil service eligibility
        if 'civil_service_eligibility' in extracted_data:
            eligibility = extracted_data['civil_service_eligibility']
            if isinstance(eligibility, list):
                for elig in eligibility:
                    if elig and elig.get('eligibility') and 'career service' in str(elig.get('eligibility', '')).lower():
                        converted_data['eligibility'].append({
                            'eligibility': elig.get('eligibility', 'N/A'),
                            'rating': elig.get('rating', 'N/A'),
                            'date_of_examination': elig.get('date_exam', 'N/A'),
                            'place_of_examination': elig.get('place_exam', 'N/A')
                        })
        
        # Voluntary work
        if 'voluntary_work' in extracted_data:
            voluntary = extracted_data['voluntary_work']
            if isinstance(voluntary, list):
                for vol in voluntary:
                    if vol and vol.get('organization'):
                        converted_data['volunteer_work'].append({
                            'organization': vol.get('organization', 'N/A'),
                            'position': vol.get('position', 'N/A'),
                            'hours': vol.get('hours', 0)
                        })
        
        # Summary
        total_entries = (len(converted_data['education']) + 
                        len(converted_data['experience']) + 
                        len(converted_data['training']) + 
                        len(converted_data['eligibility']) + 
                        len(converted_data['volunteer_work']))
        
        print(f"âœ… Conversion complete! Total entries: {total_entries}")
        print(f"   ðŸ“š Education: {len(converted_data['education'])}")
        print(f"   ðŸ’¼ Experience: {len(converted_data['experience'])}")
        print(f"   ðŸ“– Training: {len(converted_data['training'])}")
        print(f"   âœ… Eligibility: {len(converted_data['eligibility'])}")
        print(f"   ðŸ¤² Voluntary: {len(converted_data['volunteer_work'])}")
        
        return converted_data

    def _extract_personal_info_from_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Extract personal information from PDS file
        Enhanced version with better pattern matching for PDS Excel files
        """
        try:
            if filename.lower().endswith('.xlsx'):
                import pandas as pd
                print(f"ðŸ” Extracting personal info from Excel: {filename}")
                df = pd.read_excel(file_path, sheet_name='C1', header=None)
                
                personal_info = {}
                
                # Extract name components and other personal data
                for idx, row in df.iterrows():
                    if idx > 50:  # Increased range to find more data
                        break
                    
                    row_values = [str(cell) for cell in row if pd.notna(cell) and str(cell).strip() != '']
                    
                    if len(row_values) >= 3:  # Need at least 3 values
                        # Check different positions for name data
                        for i in range(len(row_values)):
                            cell_text = str(row_values[i]).upper().strip()
                            
                            # Look for surname (format: ['2.', 'SURNAME', 'COPIOSO'])
                            if 'SURNAME' in cell_text and i + 1 < len(row_values):
                                surname = str(row_values[i + 1]).strip()
                                if surname not in ['SURNAME', 'nan', '', 'N/a']:
                                    personal_info['surname'] = surname
                                    print(f"âœ… Found surname: {surname}")
                            
                            # Look for first name
                            elif 'FIRST NAME' in cell_text:
                                # Try next position first
                                if i + 1 < len(row_values):
                                    first_name = str(row_values[i + 1]).strip()
                                    if first_name not in ['FIRST NAME', 'nan', '', 'N/a', 'NAME EXTENSION']:
                                        personal_info['first_name'] = first_name
                                        print(f"âœ… Found first name: {first_name}")
                                # Also check if name is in same cell after the label
                                name_parts = cell_text.split()
                                if len(name_parts) > 2:  # "FIRST NAME ACTUALNAME"
                                    potential_name = ' '.join(name_parts[2:])
                                    if potential_name not in ['', 'NAME']:
                                        personal_info['first_name'] = potential_name
                                        print(f"âœ… Found first name in same cell: {potential_name}")
                            
                            # Look for middle name
                            elif 'MIDDLE NAME' in cell_text and i + 1 < len(row_values):
                                middle_name = str(row_values[i + 1]).strip()
                                if middle_name not in ['MIDDLE NAME', 'nan', '', 'N/a']:
                                    personal_info['middle_name'] = middle_name
                                    print(f"âœ… Found middle name: {middle_name}")
                            
                            # Look for birth date
                            elif 'DATE OF BIRTH' in cell_text and i + 1 < len(row_values):
                                birth_date = str(row_values[i + 1]).strip()
                                if birth_date not in ['DATE OF BIRTH', 'nan', '', 'N/a']:
                                    personal_info['birth_date'] = birth_date
                                    print(f"âœ… Found birth date: {birth_date}")
                            
                            # Look for place of birth
                            elif 'PLACE OF BIRTH' in cell_text and i + 1 < len(row_values):
                                birth_place = str(row_values[i + 1]).strip()
                                if birth_place not in ['PLACE OF BIRTH', 'nan', '', 'N/a']:
                                    personal_info['birth_place'] = birth_place
                                    print(f"âœ… Found birth place: {birth_place}")
                
                # Construct full name from extracted components
                if 'first_name' in personal_info or 'surname' in personal_info:
                    name_parts = []
                    if 'first_name' in personal_info:
                        name_parts.append(personal_info['first_name'])
                    if 'middle_name' in personal_info:
                        name_parts.append(personal_info['middle_name'])
                    if 'surname' in personal_info:
                        name_parts.append(personal_info['surname'])
                    
                    if name_parts:
                        full_name = ' '.join(name_parts)
                        personal_info['name'] = full_name
                        print(f"âœ… Constructed full name: {full_name}")
                
                return personal_info
            else:
                # For PDF files, try to extract from text
                return {'name': filename.replace('.pdf', '').replace('_', ' ').title()}
                
        except Exception as e:
            print(f"âš ï¸ Could not extract personal info: {e}")
            import traceback
            traceback.print_exc()
            return {'name': filename.replace('.pdf', '').replace('_', ' ').title()}


# Standalone PDS utility functions for app.py compatibility
def convert_pds_to_assessment_format(extracted_data, source_filename=None):
    """
    Convert extracted PDS data to assessment format with correct field mappings
    Standalone function for compatibility with app.py
    """
    converted_data = {
        'basic_info': extract_personal_info_manually(source_filename) if source_filename else {'name': 'Unknown Candidate'},
        'education': [],
        'experience': [],
        'experience_data': [],  # Add for assessment engine compatibility
        'training': [],
        'eligibility': [],
        'certifications': [],
        'awards': [],
        'volunteer_work': []
    }
    
    print(f"ðŸ”„ Converting PDS data...")
    
    # Educational background
    if 'educational_background' in extracted_data:
        education = extracted_data['educational_background']
        if isinstance(education, list):
            for edu in education:
                if edu and edu.get('level') and edu.get('level') not in ['N/a', '', 'nan']:
                    converted_data['education'].append({
                        'level': edu.get('level', 'N/A'),
                        'school': edu.get('school', 'N/A'),
                        'degree_course': edu.get('degree_course', 'N/A'),
                        'year_graduated': edu.get('year_graduated', 'N/A'),
                        'honors': edu.get('honors', 'N/A'),
                        'units_earned': edu.get('highest_level_units', 'N/A')
                    })
    
    # Work experience
    if 'work_experience' in extracted_data:
        experience = extracted_data['work_experience']
        if isinstance(experience, list):
            for exp in experience:
                if exp and exp.get('position'):
                    experience_entry = {
                        'position': exp.get('position', 'N/A'),
                        'company': exp.get('company', 'N/A'),
                        'from_date': exp.get('date_from', 'N/A'),
                        'to_date': exp.get('date_to', 'N/A'),
                        'monthly_salary': exp.get('salary', 'N/A'),
                        'salary_grade': exp.get('grade', 'N/A'),
                        'govt_service': 'Y' if 'government' in str(exp.get('company', '')).lower() or 'civil service' in str(exp.get('company', '')).lower() or 'deped' in str(exp.get('company', '')).lower() else 'N'
                    }
                    # Add to both fields for compatibility
                    converted_data['experience'].append(experience_entry)
                    converted_data['experience_data'].append(experience_entry)
    
    # Training and seminars
    if 'learning_development' in extracted_data:
        training = extracted_data['learning_development']
        if isinstance(training, list):
            for train in training:
                if train and train.get('title'):
                    hours = train.get('hours', 0)
                    try:
                        hours = float(hours) if hours else 0
                    except:
                        hours = 0
                    
                    converted_data['training'].append({
                        'title': train.get('title', 'N/A'),
                        'hours': hours,
                        'type': train.get('type', 'N/A'),
                        'provider': train.get('conductor', 'N/A')
                    })
    
    # Civil service eligibility
    if 'civil_service_eligibility' in extracted_data:
        eligibility = extracted_data['civil_service_eligibility']
        if isinstance(eligibility, list):
            for elig in eligibility:
                if elig and elig.get('eligibility') and 'career service' in str(elig.get('eligibility', '')).lower():
                    converted_data['eligibility'].append({
                        'eligibility': elig.get('eligibility', 'N/A'),
                        'rating': elig.get('rating', 'N/A'),
                        'date_of_examination': elig.get('date_exam', 'N/A'),
                        'place_of_examination': elig.get('place_exam', 'N/A')
                    })
    
    # Voluntary work
    if 'voluntary_work' in extracted_data:
        voluntary = extracted_data['voluntary_work']
        if isinstance(voluntary, list):
            for vol in voluntary:
                if vol and vol.get('organization'):
                    converted_data['volunteer_work'].append({
                        'organization': vol.get('organization', 'N/A'),
                        'position': vol.get('position', 'N/A'),
                        'hours': vol.get('hours', 0)
                    })
    
    # Summary
    total_entries = (len(converted_data['education']) + 
                    len(converted_data['experience']) + 
                    len(converted_data['training']) + 
                    len(converted_data['eligibility']) + 
                    len(converted_data['volunteer_work']))
    
    print(f"âœ… Conversion complete! Total entries: {total_entries}")
    print(f"   ðŸ“š Education: {len(converted_data['education'])}")
    print(f"   ðŸ’¼ Experience: {len(converted_data['experience'])}")
    print(f"   ðŸ“– Training: {len(converted_data['training'])}")
    print(f"   âœ… Eligibility: {len(converted_data['eligibility'])}")
    print(f"   ðŸ¤² Voluntary: {len(converted_data['volunteer_work'])}")
    
    return converted_data

def extract_personal_info_manually(filename=None):
    """Extract personal information from PDS Excel file - standalone function"""
    try:
        import pandas as pd
        import os
        
        # Use provided filename or default selection logic
        if not filename:
            available_files = ["Sample PDS Lenar.xlsx", "Sample PDS New.xlsx", "sample_pds.xlsx"]
            found_files = [f for f in available_files if os.path.exists(f)]
            if not found_files:
                raise FileNotFoundError("No PDS files found for personal info extraction")
            filename = found_files[0]
        
        print(f"ðŸ” Extracting personal info from: {filename}")
        df = pd.read_excel(filename, sheet_name='C1', header=None)
        
        # Default fallback (will be overridden if real data found)
        personal_info = {
            'name': 'Unknown Candidate',
            'email': 'candidate@example.com',
            'phone': 'N/A',
            'address': 'N/A',
            'citizenship': 'N/A',
            'civil_status': 'N/A',
            'birth_date': 'N/A',
            'birth_place': 'N/A'
        }
        
        # Try to extract real data
        for idx, row in df.iterrows():
            if idx > 25:
                break
            
            row_values = [str(cell) for cell in row if pd.notna(cell) and str(cell).strip() != '']
            
            # Extract name components
            if len(row_values) >= 4:
                if 'SURNAME' in str(row_values[1]).upper():
                    surname = str(row_values[3]).strip()
                    if surname not in ['SURNAME', 'nan', '']:
                        personal_info['surname'] = surname
                elif 'FIRST NAME' in str(row_values[1]).upper():
                    first_name = str(row_values[3]).strip()
                    if first_name not in ['FIRST NAME', 'nan', '']:
                        personal_info['first_name'] = first_name
                elif 'MIDDLE NAME' in str(row_values[1]).upper():
                    middle_name = str(row_values[3]).strip()
                    if middle_name not in ['MIDDLE NAME', 'nan', '']:
                        personal_info['middle_name'] = middle_name
                elif 'DATE OF BIRTH' in str(row_values[1]).upper():
                    birth_date = str(row_values[3]).strip()
                    if birth_date not in ['DATE OF BIRTH', 'nan', '']:
                        personal_info['birth_date'] = birth_date
        
        # Construct full name from extracted components
        if 'first_name' in personal_info or 'surname' in personal_info:
            name_parts = []
            if 'first_name' in personal_info:
                name_parts.append(personal_info['first_name'])
            if 'middle_name' in personal_info:
                name_parts.append(personal_info['middle_name'])
            if 'surname' in personal_info:
                name_parts.append(personal_info['surname'])
            
            if name_parts:
                full_name = ' '.join(name_parts)
                personal_info['name'] = full_name
                print(f"âœ… Constructed full name: {full_name}")
        
        return personal_info
        
    except Exception as e:
        print(f"âš ï¸ Could not extract personal info: {e}")
        return {'name': 'Unknown Candidate', 'email': 'candidate@example.com'}


# Legacy function wrappers for backward compatibility
def cleanResume(txt):
    processor = PersonalDataSheetProcessor()
    return processor.preprocess_text(txt)

def pdf_to_text(file):
    processor = PersonalDataSheetProcessor()
    return processor.extract_pdf_text(file)

def extract_contact_number_from_resume(text):
    processor = PersonalDataSheetProcessor()
    basic_info = processor.extract_basic_info(text)
    return basic_info.get('phone', '')

def extract_email_from_resume(text):
    processor = PersonalDataSheetProcessor()
    basic_info = processor.extract_basic_info(text)
    return basic_info.get('email', '')

def extract_name_from_resume(text):
    processor = PersonalDataSheetProcessor()
    basic_info = processor.extract_basic_info(text)
    return basic_info.get('name', '')

def extract_skills_from_resume(text):
    processor = PersonalDataSheetProcessor()
    return processor.extract_skills(text)

def extract_education_from_resume(text):
    processor = PersonalDataSheetProcessor()
    return processor.extract_education_pds(text)

def predict_category(text, tfidf_vectorizer, rf_classifier):
    # Legacy function - not supported in PDS system
    return "Not supported in PDS assessment system"

def job_recommendation(text, tfidf_vectorizer, rf_classifier):
    # Legacy function - not supported in PDS system
    return "Not supported in PDS assessment system"
    return processor.predict_job_recommendation(text, tfidf_vectorizer, rf_classifier)
