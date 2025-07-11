"""
Semantic Analysis Setup and Testing Script

This script helps initialize and test the BERT-based semantic analysis components.
Run this after installing the new requirements to ensure everything works correctly.
"""

import os
import sys
import logging
from utils import ResumeProcessor, SemanticAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_semantic_setup():
    """Test if semantic models are properly installed and working."""
    logger.info("Testing semantic analysis setup...")
    
    try:
        # Initialize semantic analyzer
        analyzer = SemanticAnalyzer()
        
        if analyzer.sentence_model is None:
            logger.error("Sentence transformer model failed to load")
            return False
        
        # Test basic functionality
        test_text1 = "Python programming and machine learning experience"
        test_text2 = "Software development with AI and ML skills"
        
        similarity = analyzer.semantic_similarity(test_text1, test_text2)
        logger.info(f"Semantic similarity test: {similarity:.3f}")
        
        if similarity > 0.5:
            logger.info("✓ Semantic similarity working correctly")
        else:
            logger.warning("! Semantic similarity may need adjustment")
        
        # Test skill extraction
        skills = analyzer.find_semantic_skills(
            test_text1, 
            ['python', 'machine learning', 'javascript', 'data science'],
            threshold=0.5
        )
        
        logger.info(f"Semantic skill extraction test: {skills}")
        
        if skills:
            logger.info("✓ Semantic skill extraction working")
        else:
            logger.warning("! Semantic skill extraction may need adjustment")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during semantic setup test: {str(e)}")
        return False

def test_resume_processor():
    """Test the enhanced resume processor."""
    logger.info("Testing enhanced resume processor...")
    
    try:
        processor = ResumeProcessor()
        
        sample_resume = """
        John Doe
        john.doe@email.com
        
        Senior Software Engineer with 5 years of experience in Python development,
        machine learning, and web applications. Skilled in Django, React, and AWS.
        Led a team of 3 developers on multiple projects. Experience with data analysis
        and artificial intelligence applications.
        
        Education:
        Bachelor of Computer Science, 2018
        """
        
        sample_job = """
        Looking for a Python developer with experience in AI/ML, web development,
        and cloud technologies. Leadership experience preferred. Knowledge of
        data science and software architecture required.
        """
        
        # Test enhanced skill extraction
        skills_with_context = processor.extract_skills_with_context(sample_resume)
        logger.info(f"Skills with context: {len(skills_with_context)} skills found")
        
        # Test semantic matching
        match_result = processor.match_skills_with_requirements(
            list(skills_with_context.keys()), sample_job
        )
        logger.info(f"Skills matching: {match_result['match_percentage']:.1f}% match")
        
        # Test comprehensive scoring
        score_result = processor.calculate_match_score(sample_resume, sample_job)
        logger.info(f"Comprehensive score: {score_result['final_score']:.1f}")
        
        # Test transferable skills analysis
        transferable = processor.analyze_transferable_skills(sample_resume, "data science")
        logger.info(f"Transferable skills analysis: {transferable.get('total_transferable_count', 0)} transferable skills")
        
        logger.info("✓ Enhanced resume processor working correctly")
        return True
        
    except Exception as e:
        logger.error(f"Error testing resume processor: {str(e)}")
        return False

def download_models():
    """Download required models if not already present."""
    logger.info("Checking and downloading required models...")
    
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import DistilBertTokenizer, DistilBertModel
        
        # Download sentence transformer model
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✓ Sentence transformer model loaded")
        
        # Download DistilBERT model
        logger.info("Loading DistilBERT model...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        logger.info("✓ DistilBERT model loaded")
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        return False

def main():
    """Main setup and testing function."""
    logger.info("=== ResuAI Semantic Analysis Setup ===")
    
    # Check if required packages are installed
    try:
        import torch
        import transformers
        import sentence_transformers
        import faiss
        logger.info("✓ Required packages are installed")
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False
    
    # Download models
    if not download_models():
        logger.error("Failed to download required models")
        return False
    
    # Test semantic setup
    if not test_semantic_setup():
        logger.error("Semantic analysis setup failed")
        return False
    
    # Test resume processor
    if not test_resume_processor():
        logger.error("Resume processor test failed")
        return False
    
    logger.info("=== Setup completed successfully! ===")
    logger.info("Your ResuAI system now has enhanced semantic analysis capabilities:")
    logger.info("• BERT-based semantic understanding")
    logger.info("• Improved skill extraction with context")
    logger.info("• Advanced job matching algorithms")
    logger.info("• Transferable skills analysis")
    logger.info("• Comprehensive scoring system")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
