#!/usr/bin/env python3
"""
Quick database check script
"""
from database import DatabaseManager

dm = DatabaseManager()
conn = dm.get_connection()
cursor = conn.cursor()

# Check tables
tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Database tables:")
for table in tables:
    print(f"  - {table[0]}")

# Check upload_sessions table
print("\nUpload sessions table structure:")
try:
    cursor.execute("PRAGMA table_info(upload_sessions)")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
except Exception as e:
    print(f"  Error: {e}")

# Check upload_files table  
print("\nUpload files table structure:")
try:
    cursor.execute("PRAGMA table_info(upload_files)")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
except Exception as e:
    print(f"  Error: {e}")

# Check current sessions
print("\nCurrent upload sessions:")
try:
    cursor.execute("SELECT session_id, job_id, file_count, status FROM upload_sessions ORDER BY created_at DESC LIMIT 5")
    sessions = cursor.fetchall()
    for session in sessions:
        print(f"  - {session[0]}: job_id={session[1]}, files={session[2]}, status={session[3]}")
except Exception as e:
    print(f"  Error: {e}")

conn.close()
print("\nDatabase check complete.")