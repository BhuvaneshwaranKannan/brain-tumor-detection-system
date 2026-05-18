import sqlite3
import os

db_path = "c:/Users/Vishal/OneDrive/Desktop/web 7/users.db"
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'patient'")
        conn.commit()
        conn.close()
        print("Role column added to DB successfully.")
    except Exception as e:
        print(f"Error (maybe column exists): {e}")
else:
    print("DB file not found.")
