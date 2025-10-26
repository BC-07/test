from database import DatabaseManager
dm = DatabaseManager()
conn = dm.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='upload_sessions'")
print([r['column_name'] for r in cursor.fetchall()])
conn.close()