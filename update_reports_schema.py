import sqlite3

def upgrade_db():
    print("Upgrading database schema for status and process_time...")
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        c.execute("ALTER TABLE reports ADD COLUMN status TEXT DEFAULT 'pending'")
        print("Successfully added 'status' column.")
    except sqlite3.OperationalError as e:
        print(f"Skipping status (may already exist): {e}")

    try:
        c.execute("ALTER TABLE reports ADD COLUMN process_time REAL DEFAULT 0.0")
        print("Successfully added 'process_time' column.")
    except sqlite3.OperationalError as e:
        print(f"Skipping process_time (may already exist): {e}")
        
    conn.commit()
    conn.close()
    print("Database upgrade complete.")

if __name__ == '__main__':
    upgrade_db()
