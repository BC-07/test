#!/usr/bin/env python3
"""
Test the upload and analysis fix
"""
import sys
from database import DatabaseManager

# Test the database methods
dm = DatabaseManager()

print("Testing database methods...")

# Test creating a fake session
session_id = "test-session-123"
user_id = 1
job_id = 1

print(f"1. Creating test session: {session_id}")
success = dm.create_upload_session(session_id, user_id, job_id)
print(f"   Result: {success}")

# Test storing file record
print("2. Creating test file record...")
file_data = {
    'original_name': 'test.pdf',
    'size': 1024,
    'type': 'pdf'
}
success = dm.create_upload_file_record(session_id, "test-file-123", file_data)
print(f"   Result: {success}")

# Test getting files
print("3. Getting upload files...")
files = dm.get_upload_files(session_id)
print(f"   Found {len(files)} files:")
for file in files:
    print(f"   - {file}")

# Clean up
print("4. Cleaning up test data...")
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