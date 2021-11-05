import psycopg2
import random
from tqdm import tqdm

# Generate wide table and insert data into wide table (used for real system experiments)
def generate_random_str(length):
	random_str=""
	base_str="ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789"
	n=len(base_str)-1
	for i in range(length):
		random_str+=base_str[random.randint(0,n)]

	return random_str

conn = psycopg2.connect(database="wide_test", user="postgres", password="your-password", host="127.0.0.1", port="5432")
conn.autocommit = True
cursor = conn.cursor()

# Create wide table
ddl_sql = '''
CREATE TABLE wide_table(
	a1 CHAR(150) NOT NULL,
	a2 CHAR(500) NOT NULL,
	a3 CHAR(233) NOT NULL,
	a4 CHAR(300) NOT NULL,
	a5 INTEGER NOT NULL,
	a6 CHAR(50) NOT NULL,
	a7 INTEGER NOT NULL,
	a8 CHAR(100) NOT NULL,
	a9 INTEGER NOT NULL,
	a10 INTEGER NOT NULL,
	a11 CHAR(500) NOT NULL,
	a12 CHAR(100) NOT NULL,
	a13 CHAR(250) NOT NULL,
	a14 INTEGER NOT NULL,
	a15 INTEGER NOT NULL,
	a16 CHAR(1000) NOT NULL,
	a17 CHAR(300) NOT NULL,
	a18 CHAR(25) NOT NULL,
	a19 INTEGER NOT NULL,
	a20 CHAR(400) NOT NULL,
	a21 INTEGER NOT NULL,
	a22 CHAR(33) NOT NULL,
	a23 CHAR(100) NOT NULL,
	a24 CHAR(55) NOT NULL,
	a25 CHAR(155) NOT NULL,
	a26 INTEGER NOT NULL,
	a27 INTEGER NOT NULL,
	a28 CHAR(900) NOT NULL,
	a29 CHAR(20) NOT NULL,
	a30 INTEGER NOT NULL
);
'''
cursor.execute(ddl_sql)

# Randomly generate data and insert them to wide table
for i in tqdm(range(100158)):
	sql = "INSERT INTO wide_table (a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30) VALUES ("
	
	a1 = generate_random_str(150)
	a2 = generate_random_str(500)
	a3 = generate_random_str(233)
	a4 = generate_random_str(300)
	a5 = str(random.randint(0,1000000))
	a6 = generate_random_str(50)
	a7 = str(random.randint(0,1000000))
	a8 = generate_random_str(100)
	a9 = str(random.randint(0,1000000))
	a10 = str(random.randint(0,1000000))
	a11 = generate_random_str(500)
	a12 = generate_random_str(100)
	a13 = generate_random_str(250)
	a14 = str(random.randint(0,1000000))
	a15 = str(random.randint(0,1000000))
	a16 = generate_random_str(1000)
	
	a17 = generate_random_str(300)
	a18 = generate_random_str(25)
	a19 = str(random.randint(0,1000000))
	a20 = generate_random_str(400)
	a21 = str(random.randint(0,1000000))
	a22 = generate_random_str(33)
	a23 = generate_random_str(100)
	a24 = generate_random_str(55)
	a25 = generate_random_str(155)
	a26 = str(random.randint(0,1000000))
	a27 = str(random.randint(0,1000000))
	a28 = generate_random_str(900)
	a29 = generate_random_str(20)
	a30 = str(random.randint(0,1000000))
	
	
	sql += "'"+a1+"',"
	sql += "'"+a2+"',"
	sql += "'"+a3+"',"
	sql += "'"+a4+"',"
	sql += ""+a5+","
	sql += "'"+a6+"',"
	sql += ""+a7+","
	sql += "'"+a8+"',"
	sql += ""+a9+","
	sql += ""+a10+","
	sql += "'"+a11+"',"
	sql += "'"+a12+"',"
	sql += "'"+a13+"',"
	sql += ""+a14+","
	sql += ""+a15+","
	sql += "'"+a16+"',"
	sql += "'"+a17+"',"
	sql += "'"+a18+"',"
	sql += ""+a19+","
	sql += "'"+a20+"',"
	sql += ""+a21+","
	sql += "'"+a22+"',"
	sql += "'"+a23+"',"
	sql += "'"+a24+"',"
	sql += "'"+a25+"',"
	sql += ""+a26+","
	sql += ""+a27+","
	sql += "'"+a28+"',"
	sql += "'"+a29+"',"
	sql += ""+a30+");"
	# print(sql)

	cursor.execute(sql)

cursor.close()
conn.close()