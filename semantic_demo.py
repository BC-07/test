"""
Example usage of the enhanced semantic analysis features in ResuAI

This script demonstrates the new BERT-based capabilities for better
resume analysis and job matching.
"""

from utils import ResumeProcessor, SemanticAnalyzer
import json

def example_enhanced_resume_analysis():
    """Demonstrate enhanced resume analysis capabilities."""
    print("=== Enhanced Resume Analysis Example ===\n")
    
    # Initialize processor
    processor = ResumeProcessor()
    
    # Sample resume text
    sample_resume = """
    Sarah Johnson
    sarah.johnson@email.com
    (555) 123-4567
    
    PROFESSIONAL SUMMARY
    Experienced software developer with 6 years in web development and data analytics.
    Strong background in Python programming, database management, and team leadership.
    Passionate about machine learning and artificial intelligence applications.
    
    TECHNICAL SKILLS
    • Programming: Python, JavaScript, SQL, R
    • Web Development: Django, React, HTML/CSS
    • Data Science: pandas, numpy, scikit-learn, TensorFlow
    • Databases: PostgreSQL, MongoDB, Redis
    • Cloud: AWS, Docker, Kubernetes
    • Tools: Git, Jenkins, Tableau
    
    EXPERIENCE
    Senior Data Analyst - TechCorp (2020-2023)
    • Led a team of 4 analysts in developing predictive models
    • Implemented machine learning algorithms for customer segmentation
    • Designed and maintained data pipelines using Python and SQL
    • Collaborated with cross-functional teams on product strategy
    
    Software Developer - StartupXYZ (2018-2020)
    • Developed web applications using Django and React
    • Optimized database queries improving performance by 40%
    • Mentored junior developers and conducted code reviews
    
    EDUCATION
    Master of Science in Computer Science - University of Technology (2018)
    Bachelor of Science in Mathematics - State University (2016)
    """
    
    sample_job_requirements = """
    We are seeking a Senior Machine Learning Engineer to join our AI team.
    The ideal candidate will have:
    
    Required Skills:
    - 5+ years experience in machine learning and data science
    - Proficiency in Python, TensorFlow, and scikit-learn
    - Experience with cloud platforms (AWS, Azure, or GCP)
    - Strong background in statistical analysis and modeling
    - Leadership experience managing technical teams
    - Knowledge of MLOps and model deployment
    
    Preferred Skills:
    - PhD in Computer Science, Statistics, or related field
    - Experience with deep learning frameworks
    - Knowledge of big data technologies (Spark, Hadoop)
    - Familiarity with containerization (Docker, Kubernetes)
    """
    
    print("1. BASIC INFORMATION EXTRACTION")
    print("-" * 40)
    basic_info = processor.extract_basic_info(sample_resume)
    for key, value in basic_info.items():
        print(f"{key.title()}: {value}")
    
    print("\n2. ENHANCED SKILL EXTRACTION WITH CONTEXT")
    print("-" * 40)
    skills_with_context = processor.extract_skills_with_context(sample_resume)
    print(f"Total skills found: {len(skills_with_context)}")
    
    for skill, details in list(skills_with_context.items())[:10]:  # Show first 10
        print(f"• {skill}")
        print(f"  Confidence: {details.get('confidence', 0):.2f}")
        print(f"  Experience Level: {details.get('experience_level', 'unknown')}")
    
    print("\n3. SEMANTIC SKILLS MATCHING")
    print("-" * 40)
    resume_skills = list(skills_with_context.keys())
    match_result = processor.match_skills_with_requirements(resume_skills, sample_job_requirements)
    
    print(f"Overall Match: {match_result['match_percentage']:.1f}%")
    print(f"Weighted Score: {match_result['weighted_score']:.1f}%")
    print(f"Exact Matches: {match_result['exact_match_count']}")
    print(f"Semantic Matches: {match_result['semantic_match_count']}")
    
    print("\nExact Matches:")
    for skill in match_result['exact_matches'][:5]:
        print(f"  ✓ {skill}")
    
    print("\nSemantic Matches:")
    for match in match_result['semantic_matches'][:5]:
        print(f"  ~ {match['required_skill']} → {match['matched_skill']} ({match['confidence']:.2f})")
    
    print("\nMissing Skills:")
    for skill in match_result['missing_skills'][:5]:
        print(f"  ✗ {skill}")
    
    print("\n4. COMPREHENSIVE MATCH SCORING")
    print("-" * 40)
    score_result = processor.calculate_match_score(sample_resume, sample_job_requirements)
    
    print(f"Final Score: {score_result['final_score']:.1f}/100")
    print("\nComponent Scores:")
    for component, score in score_result['component_scores'].items():
        print(f"  {component.replace('_', ' ').title()}: {score:.1f}")
    
    print("\n5. TRANSFERABLE SKILLS ANALYSIS")
    print("-" * 40)
    transferable = processor.analyze_transferable_skills(sample_resume, "artificial intelligence")
    
    print(f"Domain Relevance to AI: {transferable.get('domain_relevance', 0):.2f}")
    print(f"Total Transferable Skills: {transferable.get('total_transferable_count', 0)}")
    print(f"Strongest Category: {transferable.get('strongest_category', 'N/A')}")
    
    for category, details in transferable.get('transferable_skills', {}).items():
        print(f"\n{category.title()} Skills:")
        for skill_info in details['skills'][:3]:  # Show top 3
            print(f"  • {skill_info['skill']} (relevance: {skill_info['relevance']:.2f})")
    
    print("\n6. SEMANTIC RESUME SUMMARY")
    print("-" * 40)
    summary = processor.semantic_resume_summary(sample_resume)
    
    skill_summary = summary.get('skill_summary', {})
    experience_summary = summary.get('experience_summary', {})
    
    print(f"Skill Diversity: {skill_summary.get('skill_diversity', 0)} categories")
    print(f"Average Skill Confidence: {skill_summary.get('avg_confidence', 0):.2f}")
    print(f"Seniority Level: {experience_summary.get('seniority_level', 'unknown')}")
    print(f"Total Experiences: {experience_summary.get('total_experiences', 0)}")

def example_semantic_analyzer():
    """Demonstrate standalone semantic analyzer capabilities."""
    print("\n\n=== Semantic Analyzer Examples ===\n")
    
    analyzer = SemanticAnalyzer()
    
    print("1. SEMANTIC SIMILARITY TESTING")
    print("-" * 40)
    
    test_pairs = [
        ("machine learning", "artificial intelligence"),
        ("web development", "frontend programming"),
        ("team leadership", "project management"),
        ("python programming", "javascript development"),
        ("data analysis", "statistical modeling")
    ]
    
    for text1, text2 in test_pairs:
        similarity = analyzer.semantic_similarity(text1, text2)
        print(f"{text1} ↔ {text2}: {similarity:.3f}")
    
    print("\n2. SEMANTIC SKILL FINDING")
    print("-" * 40)
    
    text = """
    I have extensive experience in developing web applications using modern frameworks.
    My expertise includes working with machine learning algorithms and artificial intelligence
    systems. I've led several projects involving data analysis and statistical modeling.
    """
    
    skills_to_find = [
        'web development', 'machine learning', 'leadership', 'data science',
        'artificial intelligence', 'project management', 'python', 'statistics'
    ]
    
    found_skills = analyzer.find_semantic_skills(text, skills_to_find, threshold=0.4)
    
    print("Skills found with semantic analysis:")
    for skill, confidence in found_skills:
        print(f"  {skill}: {confidence:.3f}")

if __name__ == "__main__":
    print("ResuAI Enhanced Semantic Analysis Demo")
    print("=" * 50)
    
    try:
        example_enhanced_resume_analysis()
        example_semantic_analyzer()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey improvements with BERT integration:")
        print("• Better understanding of context and synonyms")
        print("• Semantic matching beyond keyword search")
        print("• Transferable skills analysis for career transitions")
        print("• Multi-component scoring for comprehensive evaluation")
        print("• Experience level detection from context")
        
    except Exception as e:
        print(f"Error running demo: {str(e)}")
        print("\nPlease ensure you have:")
        print("1. Installed all requirements: pip install -r requirements.txt")
        print("2. Run the setup script: python semantic_setup.py")
