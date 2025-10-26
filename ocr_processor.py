"""
OCR Processor Module for Resume Screening AI
Handles image preprocessing, text extraction, and confidence scoring for scanned documents
"""

import os
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import logging
import re
import json
from typing import Dict, List, Tuple, Optional
import tempfile

# Configure logging
logger = logging.getLogger(__name__)

class OCRProcessor:
    """
    Advanced OCR processor with image preprocessing and confidence scoring
    """
    
    def __init__(self):
        """Initialize OCR processor with Tesseract configuration"""
        # Set Tesseract path (adjust if needed)
        if os.name == 'nt':  # Windows
            tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            else:
                logger.warning("Tesseract not found in default location. Please ensure it's in PATH.")
        
        # OCR configuration optimized for PDS forms
        self.config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}/"\'@#$%^&*-_=+|\\`~<> \n\t'
        
        # Alternative config for better handwritten text recognition
        self.config_handwritten = '--oem 3 --psm 4 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}/"\'@#$%^&*-_=+|\\`~<> \n\t'
        
        # PDS template patterns for field extraction
        self.pds_patterns = {
            'personal_info': {
                'surname': [
                    r'(?:SURNAME|LAST\s*NAME)[:\s]*([A-Z][A-Za-z\s,.-]+)',
                    r'FAMILY\s*NAME[:\s]*([A-Z][A-Za-z\s,.-]+)'
                ],
                'first_name': [
                    r'(?:FIRST\s*NAME|GIVEN\s*NAME)[:\s]*([A-Z][A-Za-z\s,.-]+)',
                    r'FIRSTNAME[:\s]*([A-Z][A-Za-z\s,.-]+)'
                ],
                'middle_name': [
                    r'MIDDLE\s*NAME[:\s]*([A-Z][A-Za-z\s,.-]+)',
                    r'MIDDLENAME[:\s]*([A-Z][A-Za-z\s,.-]+)'
                ],
                'name_extension': [
                    r'NAME\s*EXTENSION[:\s]*([A-Z][A-Za-z\s,.-]*)',
                    r'EXTENSION[:\s]*([A-Z][A-Za-z\s,.-]*)'
                ],
                'date_of_birth': [
                    r'DATE\s*OF\s*BIRTH[:\s]*([0-9/\-\s]+)',
                    r'BIRTH\s*DATE[:\s]*([0-9/\-\s]+)',
                    r'DOB[:\s]*([0-9/\-\s]+)'
                ],
                'place_of_birth': [
                    r'PLACE\s*OF\s*BIRTH[:\s]*([A-Za-z\s,.-]+)',
                    r'BIRTH\s*PLACE[:\s]*([A-Za-z\s,.-]+)'
                ],
                'sex': [
                    r'SEX[:\s]*([MF]|MALE|FEMALE)',
                    r'GENDER[:\s]*([MF]|MALE|FEMALE)'
                ],
                'civil_status': [
                    r'CIVIL\s*STATUS[:\s]*([A-Za-z\s]+)',
                    r'MARITAL\s*STATUS[:\s]*([A-Za-z\s]+)'
                ],
                'height': [
                    r'HEIGHT[:\s]*([0-9.]+\s*(?:m|meters?|ft|feet)?)',
                    r'HT[:\s]*([0-9.]+\s*(?:m|meters?|ft|feet)?)'
                ],
                'weight': [
                    r'WEIGHT[:\s]*([0-9.]+\s*(?:kg|kgs?|lbs?|pounds?)?)',
                    r'WT[:\s]*([0-9.]+\s*(?:kg|kgs?|lbs?|pounds?)?)'
                ],
                'blood_type': [
                    r'BLOOD\s*TYPE[:\s]*([ABO]+[+-]?)',
                    r'BLOODTYPE[:\s]*([ABO]+[+-]?)'
                ],
                'gsis_id': [
                    r'GSIS\s*(?:ID\s*)?NO[:\s]*([0-9\-]+)',
                    r'GSIS[:\s]*([0-9\-]+)'
                ],
                'pag_ibig': [
                    r'PAG-IBIG\s*(?:ID\s*)?NO[:\s]*([0-9\-]+)',
                    r'PAGIBIG[:\s]*([0-9\-]+)'
                ],
                'philhealth': [
                    r'PHILHEALTH\s*(?:ID\s*)?NO[:\s]*([0-9\-]+)',
                    r'PHIL\s*HEALTH[:\s]*([0-9\-]+)'
                ],
                'sss_no': [
                    r'SSS\s*(?:ID\s*)?NO[:\s]*([0-9\-]+)',
                    r'SOCIAL\s*SECURITY[:\s]*([0-9\-]+)'
                ],
                'tin_no': [
                    r'TIN\s*(?:ID\s*)?NO[:\s]*([0-9\-]+)',
                    r'TAX\s*IDENTIFICATION[:\s]*([0-9\-]+)'
                ],
                'agency_id': [
                    r'AGENCY\s*EMPLOYEE\s*NO[:\s]*([0-9A-Z\-]+)',
                    r'EMPLOYEE\s*(?:ID\s*)?NO[:\s]*([0-9A-Z\-]+)'
                ]
            },
            'contact_info': {
                'residential_address': [
                    r'RESIDENTIAL\s*ADDRESS[:\s]*([A-Za-z0-9\s,.-]+)',
                    r'HOME\s*ADDRESS[:\s]*([A-Za-z0-9\s,.-]+)'
                ],
                'permanent_address': [
                    r'PERMANENT\s*ADDRESS[:\s]*([A-Za-z0-9\s,.-]+)',
                    r'PERM\s*ADDRESS[:\s]*([A-Za-z0-9\s,.-]+)'
                ],
                'zip_code': [
                    r'ZIP\s*CODE[:\s]*([0-9]{4})',
                    r'POSTAL\s*CODE[:\s]*([0-9]{4})'
                ],
                'telephone': [
                    r'TELEPHONE\s*NO[:\s]*([0-9\-\+\(\)\s]+)',
                    r'TEL\s*NO[:\s]*([0-9\-\+\(\)\s]+)'
                ],
                'mobile': [
                    r'MOBILE\s*NO[:\s]*([0-9\-\+\(\)\s]+)',
                    r'CELL\s*PHONE[:\s]*([0-9\-\+\(\)\s]+)',
                    r'(\+?63[0-9\-\s]{10,})',
                    r'(09[0-9\-\s]{9,})'
                ],
                'email': [
                    r'E-?MAIL\s*ADDRESS[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                    r'EMAIL[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                    r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
                ]
            },
            'education': {
                'elementary': [
                    r'ELEMENTARY[:\s]*([A-Za-z\s,.-]+)(?:YEAR\s*GRADUATED[:\s]*([0-9]{4}))?',
                    r'PRIMARY[:\s]*([A-Za-z\s,.-]+)'
                ],
                'secondary': [
                    r'SECONDARY[:\s]*([A-Za-z\s,.-]+)(?:YEAR\s*GRADUATED[:\s]*([0-9]{4}))?',
                    r'HIGH\s*SCHOOL[:\s]*([A-Za-z\s,.-]+)'
                ],
                'vocational': [
                    r'VOCATIONAL[:\s]*([A-Za-z\s,.-]+)(?:YEAR\s*GRADUATED[:\s]*([0-9]{4}))?',
                    r'TECHNICAL[:\s]*([A-Za-z\s,.-]+)'
                ],
                'college': [
                    r'COLLEGE[:\s]*([A-Za-z\s,.-]+)(?:YEAR\s*GRADUATED[:\s]*([0-9]{4}))?',
                    r'UNIVERSITY[:\s]*([A-Za-z\s,.-]+)'
                ],
                'graduate': [
                    r'GRADUATE\s*STUDIES[:\s]*([A-Za-z\s,.-]+)(?:YEAR\s*GRADUATED[:\s]*([0-9]{4}))?',
                    r'MASTER[:\s]*([A-Za-z\s,.-]+)',
                    r'DOCTORATE[:\s]*([A-Za-z\s,.-]+)'
                ]
            },
            'eligibility': [
                r'(?:CAREER\s*SERVICE|CS)[:\s]*([A-Za-z\s\-]+)(?:RATING[:\s]*([0-9.]+))?',
                r'(?:PROFESSIONAL|PROF)[:\s]*([A-Za-z\s\-]+)',
                r'ELIGIBILITY[:\s]*([A-Za-z\s\-]+)'
            ],
            'work_experience': [
                r'(?:POSITION\s*TITLE|DESIGNATION)[:\s]*([A-Za-z\s,.-]+)',
                r'(?:DEPARTMENT|AGENCY)[:\s]*([A-Za-z\s,.-]+)',
                r'(?:MONTHLY\s*SALARY|SALARY)[:\s]*([0-9,.\s]+)',
                r'(?:FROM|START)[:\s]*([0-9/\-\s]+)(?:TO|END)[:\s]*([0-9/\-\s]+)',
                r'(?:GOVT\s*SERVICE|GOV\'T)[:\s]*([YN]|YES|NO)'
            ]
        }
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess image for better OCR accuracy
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (processed_image, preprocessing_info)
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            else:
                # Handle file-like object from Flask
                image_path.seek(0)
                image_array = np.frombuffer(image_path.read(), np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not load image")
            
            original_shape = image.shape
            preprocessing_info = {
                'original_size': original_shape,
                'steps_applied': []
            }
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            preprocessing_info['steps_applied'].append('grayscale_conversion')
            
            # Noise reduction
            denoised = cv2.medianBlur(gray, 3)
            preprocessing_info['steps_applied'].append('noise_reduction')
            
            # Deskewing
            deskewed, angle = self._deskew_image(denoised)
            if abs(angle) > 0.5:
                preprocessing_info['steps_applied'].append(f'deskewing_angle_{angle:.2f}')
                gray = deskewed
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            preprocessing_info['steps_applied'].append('contrast_enhancement')
            
            # Apply Gaussian blur to reduce noise before binarization
            blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
            preprocessing_info['steps_applied'].append('gaussian_blur')
            
            # Binarization with adaptive threshold - use multiple methods and combine
            binary1 = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            binary2 = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Combine both methods by taking the better result
            binary = cv2.bitwise_and(binary1, binary2)
            preprocessing_info['steps_applied'].append('dual_adaptive_thresholding')
            
            # Morphological operations to clean up
            kernel = np.ones((1,1), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Additional morphological operations for text enhancement
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel2)
            preprocessing_info['steps_applied'].append('morphological_cleaning')
            
            # Resize if image is too small
            height, width = cleaned.shape
            if height < 300 or width < 300:
                scale_factor = max(300/height, 300/width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                preprocessing_info['steps_applied'].append(f'upscaling_{scale_factor:.2f}x')
            
            preprocessing_info['final_size'] = cleaned.shape
            
            return cleaned, preprocessing_info
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def _deskew_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct image skew
        
        Args:
            image: Grayscale image array
            
        Returns:
            Tuple of (deskewed_image, rotation_angle)
        """
        try:
            # Find edges
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 0:
                angles = []
                for line in lines[:20]:  # Use first 20 lines
                    if len(line) >= 2:  # Ensure we have both rho and theta
                        rho, theta = line[0]  # Extract from nested array
                        angle = theta * 180 / np.pi
                        # Convert to rotation angle
                        if angle > 90:
                            angle = angle - 180
                        angles.append(angle)
                
                # Calculate median angle
                if angles:
                    median_angle = np.median(angles)
                    
                    # Only correct if angle is significant
                    if abs(median_angle) > 0.5:
                        # Rotate image
                        (h, w) = image.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        rotated = cv2.warpAffine(image, M, (w, h), 
                                               flags=cv2.INTER_CUBIC, 
                                               borderMode=cv2.BORDER_REPLICATE)
                        return rotated, median_angle
            
            return image, 0.0
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {str(e)}")
            return image, 0.0
    
    def extract_text_with_confidence(self, image: np.ndarray) -> Dict:
        """
        Extract text from preprocessed image with confidence scores using multiple OCR approaches
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Dictionary with text and confidence information
        """
        try:
            # Try primary configuration first
            ocr_results = []
            
            # Method 1: Standard config
            try:
                ocr_data1 = pytesseract.image_to_data(
                    image, 
                    config=self.config, 
                    output_type=pytesseract.Output.DICT
                )
                ocr_results.append(('standard', ocr_data1))
            except Exception as e:
                logger.warning(f"Standard OCR config failed: {e}")
            
            # Method 2: Handwritten-optimized config
            try:
                ocr_data2 = pytesseract.image_to_data(
                    image, 
                    config=self.config_handwritten, 
                    output_type=pytesseract.Output.DICT
                )
                ocr_results.append(('handwritten', ocr_data2))
            except Exception as e:
                logger.warning(f"Handwritten OCR config failed: {e}")
            
            if not ocr_results:
                raise Exception("All OCR methods failed")
            
            # Choose the best result based on confidence and text length
            best_result = None
            best_score = 0
            
            for method_name, ocr_data in ocr_results:
                # Calculate metrics for this result
                text_blocks = []
                word_confidences = []
                
                for i in range(len(ocr_data['text'])):
                    text = ocr_data['text'][i].strip()
                    confidence = int(ocr_data['conf'][i])
                    
                    if text and confidence > 0:
                        text_blocks.append(text)
                        word_confidences.append(confidence)
                
                if word_confidences:
                    avg_confidence = sum(word_confidences) / len(word_confidences)
                    text_length = len(' '.join(text_blocks))
                    
                    # Score = confidence * 0.7 + (text_length/100) * 0.3
                    score = avg_confidence * 0.7 + min(text_length/100, 50) * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_result = (method_name, ocr_data, text_blocks, word_confidences)
            
            if not best_result:
                raise Exception("No valid OCR results")
            
            method_name, ocr_data, text_blocks, word_confidences = best_result
            
            # Calculate overall confidence
            if word_confidences:
                avg_confidence = sum(word_confidences) / len(word_confidences)
                min_confidence = min(word_confidences)
                max_confidence = max(word_confidences)
                
                # Weighted confidence (penalize low confidence words)
                high_conf_words = len([c for c in word_confidences if c >= 70])
                confidence_ratio = high_conf_words / len(word_confidences)
                weighted_confidence = avg_confidence * confidence_ratio
            else:
                avg_confidence = 0
                min_confidence = 0
                max_confidence = 0
                weighted_confidence = 0
            
            # Join text
            extracted_text = ' '.join(text_blocks)
            
            # Clean up text
            cleaned_text = self._clean_extracted_text(extracted_text)
            
            return {
                'text': cleaned_text,
                'raw_text': extracted_text,
                'method_used': method_name,
                'confidence': {
                    'average': round(avg_confidence, 2),
                    'weighted': round(weighted_confidence, 2),
                    'minimum': min_confidence,
                    'maximum': max_confidence,
                    'word_count': len(word_confidences),
                    'high_confidence_words': high_conf_words if word_confidences else 0
                },
                'word_confidences': word_confidences
            }
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return {
                'text': '',
                'raw_text': '',
                'method_used': 'none',
                'confidence': {
                    'average': 0,
                    'weighted': 0,
                    'minimum': 0,
                    'maximum': 0,
                    'word_count': 0,
                    'high_confidence_words': 0
                },
                'word_confidences': []
            }
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        replacements = {
            r'\b0\b': 'O',  # Zero to O
            r'\b1\b': 'I',  # One to I (context dependent)
            r'@': 'a',      # @ to a (context dependent)
            r'\|': 'l',     # | to l
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove special characters that are likely OCR errors
        text = re.sub(r'[^\w\s@.-]', ' ', text)
        
        # Normalize spacing
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_pds_fields(self, text: str) -> Dict:
        """
        Extract specific PDS fields using comprehensive pattern matching
        
        Args:
            text: Extracted text from PDS form
            
        Returns:
            Dictionary with extracted PDS fields organized by section
        """
        extracted_data = {
            'personal_info': {},
            'contact_info': {},
            'education': {},
            'eligibility': [],
            'work_experience': [],
            'government_ids': {}
        }
        
        # Extract personal information
        for field_name, patterns in self.pds_patterns['personal_info'].items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    value = match.group(1).strip()
                    if value and len(value) > 1:
                        extracted_data['personal_info'][field_name] = value
                        break
            if field_name in extracted_data['personal_info']:
                break
        
        # Extract contact information
        for field_name, patterns in self.pds_patterns['contact_info'].items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    value = match.group(1).strip()
                    if value and len(value) > 1:
                        extracted_data['contact_info'][field_name] = value
                        break
            if field_name in extracted_data['contact_info']:
                break
        
        # Extract education information
        for level_name, patterns in self.pds_patterns['education'].items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    school = match.group(1).strip()
                    year = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                    if school and len(school) > 1:
                        extracted_data['education'][level_name] = {
                            'school': school,
                            'year_graduated': year
                        }
                        break
        
        # Extract eligibility
        for pattern in self.pds_patterns['eligibility']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                eligibility = match.group(1).strip()
                rating = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                if eligibility and len(eligibility) > 1:
                    extracted_data['eligibility'].append({
                        'eligibility': eligibility,
                        'rating': rating
                    })
        
        # Extract work experience (basic extraction)
        work_exp_lines = []
        for pattern in self.pds_patterns['work_experience']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                work_exp_lines.append(match.group(0))
        
        if work_exp_lines:
            # Group related work experience information
            extracted_data['work_experience'] = self._group_work_experience(work_exp_lines)
        
        # Extract government IDs from personal info patterns
        gov_id_fields = ['gsis_id', 'pag_ibig', 'philhealth', 'sss_no', 'tin_no', 'agency_id']
        for field in gov_id_fields:
            if field in extracted_data['personal_info']:
                extracted_data['government_ids'][field] = extracted_data['personal_info'][field]
                del extracted_data['personal_info'][field]
        
        # Create a consolidated name field for backward compatibility
        personal = extracted_data['personal_info']
        name_parts = []
        if 'first_name' in personal:
            name_parts.append(personal['first_name'])
        if 'middle_name' in personal:
            name_parts.append(personal['middle_name'])
        if 'surname' in personal:
            name_parts.append(personal['surname'])
        if 'name_extension' in personal:
            name_parts.append(personal['name_extension'])
        
        if name_parts:
            extracted_data['full_name'] = ' '.join(name_parts)
        
        return extracted_data
    
    def _group_work_experience(self, work_exp_lines: List[str]) -> List[Dict]:
        """
        Group work experience lines into structured work experience entries
        
        Args:
            work_exp_lines: List of extracted work experience text lines
            
        Returns:
            List of structured work experience dictionaries
        """
        # This is a simplified grouping - in a full implementation,
        # you would use more sophisticated parsing logic
        work_experiences = []
        
        current_exp = {}
        for line in work_exp_lines:
            line_lower = line.lower()
            
            if 'position' in line_lower or 'designation' in line_lower:
                if current_exp:
                    work_experiences.append(current_exp)
                current_exp = {'position': line.split(':')[-1].strip()}
            elif 'department' in line_lower or 'agency' in line_lower:
                current_exp['department'] = line.split(':')[-1].strip()
            elif 'salary' in line_lower:
                current_exp['salary'] = line.split(':')[-1].strip()
            elif 'from' in line_lower and 'to' in line_lower:
                current_exp['period'] = line.split(':')[-1].strip()
            elif 'govt' in line_lower or "gov't" in line_lower:
                current_exp['govt_service'] = line.split(':')[-1].strip()
        
        if current_exp:
            work_experiences.append(current_exp)
        
        return work_experiences
    
    def process_pds_image(self, image_path, job_id: int = None) -> Dict:
        """
        Complete PDS processing pipeline for image files with enhanced template recognition
        
        Args:
            image_path: Path to image file or file object
            job_id: Optional job ID for scoring
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Preprocess image
            processed_image, preprocessing_info = self.preprocess_image(image_path)
            
            # Extract text with confidence
            ocr_result = self.extract_text_with_confidence(processed_image)
            
            # Extract PDS fields using enhanced patterns
            pds_fields = self.extract_pds_fields(ocr_result['text'])
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(ocr_result['confidence']['weighted'])
            
            logger.info(f"OCR completed using {ocr_result.get('method_used', 'unknown')} method - Weighted confidence: {ocr_result['confidence']['weighted']}%")
            logger.info(f"Extracted {len(pds_fields)} PDS sections with {confidence_level} confidence level")
            
            # Create name from structured data or fallback
            candidate_name = pds_fields.get('full_name', 'Unknown OCR Candidate')
            if candidate_name == 'Unknown OCR Candidate':
                # Try to construct from parts
                personal_info = pds_fields.get('personal_info', {})
                name_parts = []
                for key in ['first_name', 'middle_name', 'surname']:
                    if key in personal_info:
                        name_parts.append(personal_info[key])
                if name_parts:
                    candidate_name = ' '.join(name_parts)
            
            # Get contact information
            contact_info = pds_fields.get('contact_info', {})
            email = contact_info.get('email', '')
            phone = contact_info.get('mobile', contact_info.get('telephone', ''))
            address = contact_info.get('residential_address', 
                                     contact_info.get('permanent_address', ''))
            
            # Format education for storage
            education_list = []
            for level, edu_data in pds_fields.get('education', {}).items():
                if isinstance(edu_data, dict):
                    education_list.append({
                        'level': level,
                        'school': edu_data.get('school', ''),
                        'year': edu_data.get('year_graduated', '')
                    })
                else:
                    education_list.append({
                        'level': level,
                        'school': str(edu_data),
                        'year': ''
                    })
            
            # Prepare candidate data
            candidate_data = {
                'name': candidate_name,
                'email': email,
                'phone': phone,
                'address': address,
                'resume_text': ocr_result['text'],
                'education': json.dumps(education_list),
                'work_experience': json.dumps(pds_fields.get('work_experience', [])),
                'skills': json.dumps([]),
                'certifications': json.dumps([]),
                'category': 'Unknown',
                'score': 0,
                'status': 'pending',
                'processing_type': 'ocr_scanned',
                'ocr_confidence': ocr_result['confidence']['weighted'],
                'pds_data': json.dumps({
                    'personal_info': pds_fields.get('personal_info', {}),
                    'contact_info': pds_fields.get('contact_info', {}),
                    'education': pds_fields.get('education', {}),
                    'eligibility': pds_fields.get('eligibility', []),
                    'work_experience': pds_fields.get('work_experience', []),
                    'government_ids': pds_fields.get('government_ids', {}),
                    'ocr_confidence': ocr_result['confidence'],
                    'preprocessing_info': preprocessing_info,
                    'template_recognition': {
                        'fields_extracted': len([k for section in pds_fields.values() 
                                               for k in (section.keys() if isinstance(section, dict) else [])]),
                        'sections_found': list(pds_fields.keys()),
                        'confidence_level': confidence_level
                    }
                })
            }
            
            if job_id:
                candidate_data['job_id'] = job_id
            
            return {
                'success': True,
                'candidate_data': candidate_data,
                'ocr_result': ocr_result,
                'pds_fields': pds_fields,
                'confidence_level': confidence_level,
                'preprocessing_info': preprocessing_info,
                'template_analysis': {
                    'sections_detected': list(pds_fields.keys()),
                    'personal_info_completeness': len(pds_fields.get('personal_info', {})) / 16,  # 16 expected fields
                    'contact_info_completeness': len(pds_fields.get('contact_info', {})) / 6,     # 6 expected fields
                    'education_levels_found': len(pds_fields.get('education', {})),
                    'work_experience_entries': len(pds_fields.get('work_experience', [])),
                    'eligibility_entries': len(pds_fields.get('eligibility', []))
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing PDS image: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'candidate_data': None
            }
    
    def _determine_confidence_level(self, confidence_score: float) -> str:
        """
        Determine confidence level based on score
        
        Args:
            confidence_score: Weighted confidence score
            
        Returns:
            Confidence level string
        """
        if confidence_score >= 85:
            return 'high'
        elif confidence_score >= 70:
            return 'medium'
        elif confidence_score >= 50:
            return 'low'
        else:
            return 'very_low'
    
    def batch_process_images(self, image_files: List, job_id: int = None) -> List[Dict]:
        """
        Process multiple images in batch
        
        Args:
            image_files: List of image file paths or objects
            job_id: Optional job ID
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, image_file in enumerate(image_files):
            try:
                logger.info(f"Processing image {i+1}/{len(image_files)}")
                result = self.process_pds_image(image_file, job_id)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {str(e)}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'candidate_data': None
                })
        
        return results
    
    def get_confidence_color(self, confidence_score: float) -> str:
        """
        Get color code for confidence score display
        
        Args:
            confidence_score: Confidence score (0-100)
            
        Returns:
            Color class name
        """
        if confidence_score >= 85:
            return 'success'  # Green
        elif confidence_score >= 70:
            return 'warning'  # Yellow
        elif confidence_score >= 50:
            return 'info'     # Blue
        else:
            return 'danger'   # Red
