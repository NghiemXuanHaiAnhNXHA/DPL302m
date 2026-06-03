import sqlite3

DB_PATH = "potholes.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT * FROM potholes")
rows = cursor.fetchall()
c=0
for row in rows:
    print(row)
    c+=1
print(c)
conn.close()
