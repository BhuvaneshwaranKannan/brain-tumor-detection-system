import sqlite3

db_path = "c:/Users/Vishal/OneDrive/Desktop/web 7/users.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS reports(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT,
    doctor_email TEXT,
    result TEXT,
    tumor_type TEXT,
    stage TEXT,
    image_path TEXT,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()
conn.close()
print("Reports table created.")
