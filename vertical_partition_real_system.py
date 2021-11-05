import time
import dataset
import VPGAE
import column
import row
import optimal
import hillclimb
import psycopg2
import copy

import my_cost_model

import numpy as np

from workload_class import VPGAE_Workload, Workload

wide_table_attrs = 	[["a1","CHAR(150) NOT NULL"],
					 ["a2","CHAR(500) NOT NULL"],
					 ["a3","CHAR(233) NOT NULL"],
					 ["a4","CHAR(300) NOT NULL"],
					 ["a5","INTEGER NOT NULL"],
					 ["a6","CHAR(50) NOT NULL"],
					 ["a7","INTEGER NOT NULL"],
					 ["a8","CHAR(100) NOT NULL"],
					 ["a9","INTEGER NOT NULL"],
					 ["a10","INTEGER NOT NULL"],
					 ["a11","CHAR(500) NOT NULL"],
					 ["a12","CHAR(100) NOT NULL"],
					 ["a13","CHAR(250) NOT NULL"],
					 ["a14","INTEGER NOT NULL"],
					 ["a15","INTEGER NOT NULL"],
					 ["a16","CHAR(1000) NOT NULL"],
					 ["a17","CHAR(300) NOT NULL"],
					 ["a18","CHAR(25) NOT NULL"],
					 ["a19","INTEGER NOT NULL"],
					 ["a20","CHAR(400) NOT NULL"],
					 ["a21","INTEGER NOT NULL"],
					 ["a22","CHAR(33) NOT NULL"],
					 ["a23","CHAR(100) NOT NULL"],
					 ["a24","CHAR(55) NOT NULL"],
					 ["a25","CHAR(155) NOT NULL"],
					 ["a26","INTEGER NOT NULL"],
					 ["a27","INTEGER NOT NULL"],
					 ["a28","CHAR(900) NOT NULL"],
					 ["a29","CHAR(20) NOT NULL"],
					 ["a30","INTEGER NOT NULL"]]

# If list1 include all elements in list2
def include(list1,list2):
	for ele in list2:
		if(ele not in list1):
			return False
	return True

def doPartitioningOnLineitemWithoutJoin(partitions,workload):
	subtables = ["wide_table"+str(i) for i in range(len(partitions))]

	conn = psycopg2.connect(database="wide_test", user="postgres", password="your-password", host="127.0.0.1", port="5432")
	conn.autocommit = True
	cursor = conn.cursor()

	st = time.time()
	temp_partitions = copy.deepcopy(partitions)
	
	
	# Create subtables 
	for index,subtable in enumerate(subtables):
		print(subtable+":")
		for attrid in temp_partitions[index]:
			print(wide_table_attrs[attrid-1][0])

		sql = "create table "+subtable+" ("

		for attrid in temp_partitions[index]:
			sql += wide_table_attrs[attrid-1][0] + " " + wide_table_attrs[attrid-1][1] + ","

		sql = sql + "tuple_id INTEGER NOT NULL GENERATED ALWAYS AS IDENTITY ( INCREMENT 1 START 1 ))"
		cursor.execute(sql)

	# If there is clustered index
	if workload.cluster_index != None:
		for index,partition in enumerate(temp_partitions):
			# If there is a primary partition
			if include(partition,workload.cluster_index):
				sql = "alter table "+"wide_table"+str(index)+" add constraint wide_table"+str(index)+"_pkey primary key ("
				for attrid in workload.cluster_index:
					sql+="a"+str(attrid)+","
				sql=sql[:-1]
				sql+=");"
				print(sql)
				cursor.execute(sql)

	# Insert data into subtables
	for idx,subtable in enumerate(subtables):
		sql = "insert into " + subtable + " select "
		for attrid in temp_partitions[idx]:
			sql += wide_table_attrs[attrid-1][0] + ","
		sql = sql[:-1]
		
		sql += " from wide_table"
		cursor.execute(sql)
	print("layout time: {}".format(time.time()-st))
	

	sql_list = []
	# Execute all queries in the workload
	for index,query_attributes in enumerate(workload.required_attributes):
		# Find all subtables that are referenced in the query
		dict_ = dict()

		required_subtables_id = set()
		for idx,partition in enumerate(partitions):
			for attrid in query_attributes:
				if attrid in partition:
					if idx in list(dict_.keys()):
						dict_[idx].append(attrid)
					else:
						dict_[idx] = [attrid]
		temp_sql_list = []
		for partitionid,attr_list in dict_.items():
			sql = "explain analyse select "
			for attr in attr_list:
				sql += wide_table_attrs[attr-1][0] + ","
			sql = sql[:-1]
			sql += " from " +  "wide_table" + str(partitionid)
			temp_sql_list.append(sql)
					
		for i in range(int(workload.freq[index])):
			for temp_sql in temp_sql_list:
				sql_list.append(temp_sql)
		print(temp_sql_list)
		print("frequency = "+str(int(workload.freq[index])))
		print("----------------------------")

	print("start cache warm-up.")
	# cache warm-up
	for sql in sql_list:
		cursor.execute(sql)
		# print(cursor.fetchall())

	for sql in sql_list:
		cursor.execute(sql)
		# print(cursor.fetchall())

	for sql in sql_list:
		cursor.execute(sql)
		# print(cursor.fetchall())
	for sql in sql_list:
		cursor.execute(sql)
		# print(cursor.fetchall())
	for sql in sql_list:
		cursor.execute(sql)
		# print(cursor.fetchall())
	
	print("end cache warm-up.")

	st = time.time()
	for sql in sql_list:
		cursor.execute(sql)
		# print(cursor.fetchall())
	print("execution time of workload on partitioned tables: {}".format(time.time()-st))
	
	
	# Delete all subtables
	for subtable in subtables:
		sql = "drop table " + subtable + ";"
		cursor.execute(sql)
	
	cursor.close()
	conn.close()


def convert2sql(data):
	origin_sqls = []

	for idx in range(data[0]):
		sql = "explain analyse select "
		for attrid in data[2][idx]:
			sql += wide_table_attrs[attrid-1][0] + ","
		sql = sql[:-1]
		sql += " from wide_table"
		for i in range(data[3][idx]):
			origin_sqls.append(sql)
		
	return origin_sqls
'''
def execution_sqls(sqls):
	conn = psycopg2.connect(database="wide_test", user="postgres", password="your-password", host="127.0.0.1", port="5432")
	conn.autocommit = True
	cursor = conn.cursor()

	fetch_time=0
	for sql in sqls:
		cursor.execute(sql)
		# print(cursor.fetchall())
		
	cursor.close()
	conn.close()
'''

if __name__ == "__main__":
	data = dataset.real_system_wide_table()
	workload = Workload(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8])
	vpgae_workload = VPGAE_Workload(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8])

	# cost, partitions = VPGAE.partition(algo_type="VPGAE-B",workload=vpgae_workload,n_hid=32,n_dim=16,k=3,origin_candidate_length=3,beam_search_width=3)
	# cost, partitions = VPGAE.partition(algo_type="VPGAE",workload=vpgae_workload,n_hid=32,n_dim=16,k=3)
	# cost, partitions = hillclimb.partition(workload=workload)
	# cost, partitions = column.partition(workload=workload)
	# cost, partitions = row.partition(workload=workload)

	# HYRISE partitioning_scheme, estimated cost = 53844
	# partitions = [[10, 9, 8],[11],[7, 6],[5],[4],[24, 16, 15, 14, 2],[13, 12],[1],[3],[27],[21, 20, 17],[23, 19, 18],[30, 29, 28, 26, 25],[22]]
	# cost = my_cost_model.calculate_cost_fair(partitions,workload)

	# NAVATHE partitioning_scheme, estimated cost = 55279
	# partitions = [[23, 4],[13, 12],[7, 6],[11, 5],[10, 9, 8],[3],[24, 16, 15, 14, 2],[21, 20, 17],[22],[30, 29, 28, 26, 25],[27],[19, 18],[1]]
	# cost = my_cost_model.calculate_cost_fair(partitions,workload)
	
	# O2P partitioning_scheme, estimated cost = 55946
	partitions = [[23, 4],[12],[13],[6],[7],[11],[5],[8],[9],[10],[3],[2],[14],[15],[16],[24],[17],[20],[21],[22],[25],[26],[28],[29],[30],[27],[19,18],[1]]
	cost = my_cost_model.calculate_cost_fair(partitions,workload)

	partitions=sorted(partitions,key=lambda x:min(x))
	
	doPartitioningOnLineitemWithoutJoin(partitions,workload)
	print("Estimated cost:", cost)