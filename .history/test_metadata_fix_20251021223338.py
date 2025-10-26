#!/usr/bin/env python3
"""
Test the full upload with metadata
"""
import json
import sys
from database import DatabaseManager

# Test the database methods with metadata
dm = DatabaseManager()

print("Testing full upload with metadata...")

# Test creating a session with metadata
session_id = "test-session-456"
user_id = 1
job_id = 1

print(f"1. Creating test session: {session_id}")
success = dm.create_upload_session(session_id, user_id, job_id)
print(f"   Result: {success}")

# Test storing file record
print("2. Creating test file record...")
file_data = {
    'original_name': 'test_resume.pdf',
    'size': 2048,
    'type': 'pdf'
}
file_id = "test-file-456"
success = dm.create_upload_file_record(session_id, file_id, file_data)
print(f"   Result: {success}")

# Update session with metadata (simulating what upload_files_clean does)
print("3. Adding metadata to session...")
file_metadata = {
    file_id: {
        'temp_path': '/temp/test_resume.pdf',
        'original_name': 'test_resume.pdf',
        'file_id': file_id
    }
}

session_data = {
    'job_info': {'id': job_id, 'title': 'Test Job'},
    'file_metadata': file_metadata
}

success = dm.update_upload_session(session_id, 
    file_count=1,
    session_data=json.dumps(session_data)
)
print(f"   Metadata update result: {success}")

# Test getting files with metadata
print("4. Getting upload files with metadata...")
files = dm.get_upload_files(session_id)
print(f"   Found {len(files)} files:")
for file in files:
    print(f"   - {file}")
    
    # Check if it has the required fields for analysis
    required_fields = ['file_id', 'temp_path', 'original_name']
    missing_fields = [field for field in required_fields if field not in file]
    if missing_fields:
        print(f"     ⚠ Missing fields: {missing_fields}")
    else:
        print(f"     ✓ All required fields present")

# Clean up
print("5. Cleaning up test data...")
try:
    conn = dm.get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM upload_files WHERE session_id = %s", (session_id,))
    cursor.execute("DELETE FROM upload_sessions WHERE session_id = %s", (session_id,))
    conn.commit()
    conn.close()
    print("   Cleanup successful")
except Exception as e:
    print(f"   Cleanup error: {e}")

print("Test complete!")