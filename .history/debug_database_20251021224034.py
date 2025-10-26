#!/usr/bin/env python3
"""
Quick database check script
"""
from database import DatabaseManager

dm = DatabaseManager()
conn = dm.get_connection()
cursor = conn.cursor()

# Check tables (PostgreSQL)
cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
tables = cursor.fetchall()
print("Database tables:")
for table in tables:
    print(f"  - {table['table_name']}")

# Check upload_sessions table
print("\nUpload sessions table structure:")
try:
    cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='upload_sessions'")
    columns = cursor.fetchall()
    if columns:
        for col in columns:
            print(f"  - {col['column_name']} ({col['data_type']})")
    else:
        print("  Table 'upload_sessions' does not exist")
except Exception as e:
    print(f"  Error: {e}")

# Check upload_files table  
print("\nUpload files table structure:")
try:
    cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='upload_files'")
    columns = cursor.fetchall()
    if columns:
        for col in columns:
            print(f"  - {col['column_name']} ({col['data_type']})")
    else:
        print("  Table 'upload_files' does not exist")
except Exception as e:
    print(f"  Error: {e}")

# Check current sessions
print("\nCurrent upload sessions:")
try:
    cursor.execute("SELECT session_id, job_id, file_count, status FROM upload_sessions ORDER BY created_at DESC LIMIT 5")
    sessions = cursor.fetchall()
    if sessions:
        for session in sessions:
            print(f"  - {session['session_id']}: job_id={session['job_id']}, files={session['file_count']}, status={session['status']}")
    else:
        print("  No upload sessions found")
except Exception as e:
    print(f"  Error: {e}")

conn.close()
print("\nDatabase check complete.")