import sqlite3

def upgrade_db():
    print("Upgrading database schema...")
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        c.execute('ALTER TABLE reports ADD COLUMN confidence REAL DEFAULT 0.0')
        print("Successfully added 'confidence' column.")
    except sqlite3.OperationalError as e:
        print(f"Skipping confidence (may already exist): {e}")
        
    try:
        c.execute('ALTER TABLE reports ADD COLUMN slice_index INTEGER DEFAULT 0')
        print("Successfully added 'slice_index' column.")
    except sqlite3.OperationalError as e:
        print(f"Skipping slice_index (may already exist): {e}")
        
    conn.commit()
    conn.close()
    print("Database upgrade complete.")

if __name__ == '__main__':
    upgrade_db()
