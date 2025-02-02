import psycopg2

conn = psycopg2.connect(
    database='daily_summary_db',
    user='postgres',
    host='localhost',
    password='123456',
    port='5432'
)

cursor = conn.cursor()
cursor.execute('SELECT * FROM summary6')
rows = cursor.fetchall()
for row in rows:
    print(row)