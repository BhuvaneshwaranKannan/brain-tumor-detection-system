import sqlite3

conn = sqlite3.connect("users.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM users")

rows = cursor.fetchall()

if len(rows) == 0:
    print("No users found")
else:
    for row in rows:
        print(row)