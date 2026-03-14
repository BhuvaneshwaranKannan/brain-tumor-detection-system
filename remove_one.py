import sqlite3

user_id = int(input("Enter user serial number (id) to delete: "))

conn = sqlite3.connect("users.db")
cursor = conn.cursor()

cursor.execute("DELETE FROM users WHERE id=?", (user_id,))

conn.commit()

if cursor.rowcount > 0:
    print("User deleted successfully")
else:
    print("User not found")

conn.close()