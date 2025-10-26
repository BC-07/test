#!/usr/bin/env python3
"""
Check exact database schema for upload_files
"""
from database import DatabaseManager

dm = DatabaseManager()
conn = dm.get_connection()
cursor = conn.cursor()

print("upload_files table exact structure:")
cursor.execute("SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_name='upload_files' ORDER BY ordinal_position")
columns = cursor.fetchall()
for col in columns:
    print(f"  {col['column_name']}: {col['data_type']} {'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'} {col['column_default'] or ''}")

conn.close()