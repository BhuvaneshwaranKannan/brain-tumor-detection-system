import sqlite3
from werkzeug.security import generate_password_hash

def secure_passwords():
    print("Securing user passwords by hashing them...")
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute("SELECT id, password FROM users")
    users = c.fetchall()
    
    updated = 0
    for user_id, password in users:
        if not password.startswith("scrypt:") and not password.startswith("pbkdf2:"):
            # Plain text, needs hashing
            hashed = generate_password_hash(password)
            c.execute("UPDATE users SET password=? WHERE id=?", (hashed, user_id))
            updated += 1
            
    conn.commit()
    conn.close()
    print(f"Secured {updated} plain text passwords.")

if __name__ == '__main__':
    secure_passwords()
