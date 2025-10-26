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
        lines = f.readlines()
    
    fixed = False
    for i, line in enumerate(lines):
        if 'process_pdf_file(file_path, filename, job)' in line and 'Excel' in lines[max(0, i-2)]:
            print(f"📍 Found problematic line {i+1}: {line.strip()}")
            # Replace with the correct method call
            lines[i] = line.replace('process_pdf_file', 'process_excel_pds_file')
            print(f"✅ Fixed to: {lines[i].strip()}")
            fixed = True
            break
    
    if fixed:
        with open('app.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("✅ app.py fixed successfully!")
        return True
    else:
        print("❌ Could not find the problematic line")
        return False

if __name__ == "__main__":
    fix_app_py()