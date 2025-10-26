#!/usr/bin/env python3
"""
Check PostgreSQL tables
"""
from database import DatabaseManager

dm = DatabaseManager()
conn = dm.get_connection()
cursor = conn.cursor()

print("Checking PostgreSQL tables...")
cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")
tables = cursor.fetchall()

print(f"Found {len(tables)} tables:")
for table in tables:
    print(f"  - {table['table_name']}")

# Check specifically for lspu_job_postings
print("\nChecking for lspu_job_postings table...")
if any(table['table_name'] == 'lspu_job_postings' for table in tables):
    print("✓ lspu_job_postings table exists")
    
    # Get some sample data
    cursor.execute("SELECT COUNT(*) as count FROM lspu_job_postings")
    count = cursor.fetchone()
    print(f"  Records: {count['count']}")
    
    if count['count'] > 0:
        cursor.execute("SELECT id, position_title FROM lspu_job_postings LIMIT 5")
        jobs = cursor.fetchall()
        print("  Sample jobs:")
        for job in jobs:
            print(f"    ID {job['id']}: {job['position_title']}")
else:
    print("✗ lspu_job_postings table missing")

# Check legacy jobs table
print("\nChecking for legacy jobs table...")
if any(table['table_name'] == 'jobs' for table in tables):
    print("✓ jobs table exists")
    
    cursor.execute("SELECT COUNT(*) as count FROM jobs")
    count = cursor.fetchone()
    print(f"  Records: {count['count']}")
    
    if count['count'] > 0:
        cursor.execute("SELECT id, title FROM jobs LIMIT 5")
        jobs = cursor.fetchall()
        print("  Sample jobs:")
        for job in jobs:
            print(f"    ID {job['id']}: {job['title']}")
else:
    print("✗ jobs table missing")

conn.close()
print("\nTable check complete.")