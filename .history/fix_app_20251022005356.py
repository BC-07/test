#!/usr/bin/env python3
"""
Fix the specific line in app.py that calls the wrong method for Excel processing
"""

import os

def fix_app_py():
    """Fix the Excel processing line in app.py"""
    print("🔧 FIXING APP.PY EXCEL PROCESSING")
    print("=" * 40)
    
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Show context around the problematic lines
    lines = content.splitlines()
    
    # Look for Excel processing method around line 2695
    fixed = False
    for i in range(2690, min(2700, len(lines))):
        if i < len(lines):
            line = lines[i]
            print(f"Line {i+1}: {line}")
            
            # Look for the specific pattern: Excel processing calling process_pdf_file
            if ('process_pdf_file(file_path, filename, job)' in line and 
                i > 0 and 'Excel' in lines[i-1]):
                print(f"📍 Found problematic line {i+1}: {line}")
                # Replace with the correct method call
                lines[i] = line.replace('process_pdf_file', 'process_excel_pds_file')
                print(f"✅ Fixed to: {lines[i]}")
                fixed = True
    
    if fixed:
        # Write back the fixed content
        with open('app.py', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print("✅ app.py fixed successfully!")
        return True
    else:
        print("❌ Could not find the problematic line")
        print("Let me search more broadly...")
        
        # Search for all process_pdf_file calls
        for i, line in enumerate(lines):
            if 'self.pds_processor.process_pdf_file(file_path, filename, job)' in line:
                print(f"Found process_pdf_file call at line {i+1}")
                # Check context to see if this is in Excel processing
                for j in range(max(0, i-5), min(len(lines), i+2)):
                    print(f"  {j+1}: {lines[j]}")
                
                # If we find Excel context, fix it
                context = '\n'.join(lines[max(0, i-3):i+1]).lower()
                if 'excel' in context:
                    print(f"📍 Found Excel context at line {i+1}")
                    lines[i] = line.replace('process_pdf_file', 'process_excel_pds_file')
                    print(f"✅ Fixed to: {lines[i]}")
                    
                    # Write back the fixed content
                    with open('app.py', 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                    print("✅ app.py fixed successfully!")
                    return True
        
        return False

if __name__ == "__main__":
    fix_app_py()