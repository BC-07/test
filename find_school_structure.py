"""
Find where school names are actually located in the PDF structure
"""

from improved_pds_extractor import ImprovedPDSExtractor
import re

def find_school_locations():
    print("=== FINDING SCHOOL NAME LOCATIONS IN PDF ===")
    
    extractor = ImprovedPDSExtractor()
    text = extractor._extract_pdf_text('SamplePDS_Merged.pdf')
    
    # Look for "NAME OF SCHOOL" section which should contain the actual school names
    name_of_school_pattern = r'NAME\s+OF\s+SCHOOL.*?(?=BASIC|DEGREE|PERIOD|YEAR|HONORS|$)'
    
    matches = re.finditer(name_of_school_pattern, text, re.IGNORECASE | re.DOTALL)
    
    for i, match in enumerate(matches):
        print(f"\nMatch {i+1} - NAME OF SCHOOL section:")
        print("=" * 60)
        section_text = match.group()
        print(section_text[:500])
        print("=" * 60)
        
        # Extract potential school names from this section
        lines = section_text.split('\n')
        potential_schools = []
        
        for line in lines:
            line = line.strip()
            # Skip header lines and empty lines
            if (line and 
                not line.upper().startswith('NAME') and 
                not line.upper().startswith('(WRITE') and
                len(line) > 3 and
                not line.isdigit() and
                not re.match(r'^\d{2,4}$', line)):  # Skip years
                
                # Look for lines that contain school-like words
                school_indicators = ['school', 'university', 'college', 'institute', 'academy', 'center']
                if (any(indicator in line.lower() for indicator in school_indicators) or
                    len(line.split()) >= 2):  # Multi-word names are likely schools
                    potential_schools.append(line)
        
        print(f"Potential schools found: {potential_schools}")
    
    # Also search for the structure around educational levels
    print("\n" + "="*60)
    print("ANALYZING EDUCATION LEVEL STRUCTURE")
    print("="*60)
    
    # Look for the table structure with levels and school names
    education_table_pattern = r'LEVEL.*?NAME\s+OF\s+SCHOOL.*?(?=IV\.|CIVIL\s+SERVICE|$)'
    table_match = re.search(education_table_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if table_match:
        table_text = table_match.group()
        print("Education table structure found:")
        print(table_text[:800])
        
        # Try to parse the tabular data
        lines = table_text.split('\n')
        current_level = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line contains an education level
            levels = ['ELEMENTARY', 'SECONDARY', 'VOCATIONAL', 'COLLEGE', 'GRADUATE']
            for level in levels:
                if level in line.upper():
                    current_level = level
                    print(f"\nFound level: {level}")
                    
                    # Look for school name in the same line or nearby lines
                    # Remove the level keyword and extract remaining text
                    remaining = re.sub(level, '', line, flags=re.IGNORECASE).strip()
                    if remaining and len(remaining) > 3:
                        print(f"Potential school for {level}: {remaining}")
                    break

if __name__ == "__main__":
    find_school_locations()