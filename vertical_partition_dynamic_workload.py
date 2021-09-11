import time
import random
import math
import dataset
import VPGAE
import column
import row
import optimal
import hillclimb
import my_cost_model

import numpy as np

from workload_class import Workload

def merge_queries2workload(current_query_list):
	data=[]
	data.append(len(current_query_list))
	data.append(current_query_list[0][1])
	data.append([])
	data.append([])
	data.append([])
	data.append([])
	data.append([])
	data[5] = current_query_list[0][5]

	for query in current_query_list:
		data[2] += query[2]
		data[3] += query[3]
		data[4] += query[4]
		data[6] += query[6]
	data.append(current_query_list[0][7])
	data.append(current_query_list[0][8])

	return Workload(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8])

'''
# Real situation
if __name__ == "__main__":
	query_datasets, static_data = dataset.dynamic_workloads(100,1)
	dynamic_cost = 0
	static_cost = 0

	current_query_list = []
	finished_query_list = []
	time_interval_list = [random.random() for _ in range(len(query_datasets))]
	query_arrive_time_list = []
	query_finish_time_list = []
	# initialize threshold t = 2
	t=2

	for i in range(len(time_interval_list)):
		temp=0
		for j in range(i+1):
			temp += time_interval_list[j]
		query_arrive_time_list.append(temp)

	for i in range(len(query_arrive_time_list)):
		query_finish_time_list.append(query_arrive_time_list[i]+random.random()*5)

	st=time.time()
	old_time = 0.0
	finished_query_num = 0
	# Pressume that the table is Row layout at the beginning
	partitioning_scheme = [[i+1 for i in range(static_data[1])]]

	while(True):
		current_time=time.time()-st
		new_queries_are_submitted = False
		# Add new queries to workload
		for i in range(len(query_arrive_time_list)):
			if old_time <= query_arrive_time_list[i] and query_arrive_time_list[i] < current_time:
				current_query_list.append(query_datasets[i])
				new_queries_are_submitted = True
				print("query {} arrived.".format(i))
		# Collect finished queries
		for i in range(len(query_finish_time_list)):
			if old_time <= query_finish_time_list[i] and query_finish_time_list[i] < current_time:
				finished_query_list.append(query_datasets[i])
				print("query {} finished.".format(i))
				finished_query_num += 1
		# All queries are finished, break while
		if finished_query_num == len(query_datasets):
			break

		# A new query is submitted
		if new_queries_are_submitted:
			current_workload = merge_queries2workload(current_query_list)
			t = max(2,math.ceil(0.25*current_workload.query_num))
			if my_cost_model.calculate_cost_fair(partitioning_scheme,current_workload)>my_cost_model.calculate_cost_fair(column.partition(current_workload)[1],current_workload):
				_,partitioning_scheme = VPGAE.partition(algo_type="VPGAE-B",workload=current_workload,n_hid=32,n_dim=16,k=3,origin_candidate_length=3,beam_search_width=1)
		
		# When the length of finished_query_list is more than t, examine whether we have to update the partitioning scheme.
		if len(finished_query_list) >= t:
			# Remove finished queries from workload
			for finished_query in finished_query_list:
				for query in current_query_list:
					if query == finished_query:
						current_query_list.remove(query)
			t = max(2,math.ceil(0.25*current_workload.query_num))
			if my_cost_model.calculate_cost_fair(partitioning_scheme,current_workload)>my_cost_model.calculate_cost_fair(column.partition(current_workload)[1],current_workload):
				_,partitioning_scheme = VPGAE.partition(algo_type="VPGAE-B",workload=current_workload,n_hid=32,n_dim=16,k=3,origin_candidate_length=3,beam_search_width=1)
			finished_query_list = []
		old_time=current_time

	
	static_workload = Workload(static_data[0],static_data[1],static_data[2],static_data[3],static_data[4],static_data[5],static_data[6],static_data[7],static_data[8])
	kmeans_cost, kmeans_partitions = VPGAE.partition(algo_type="VPGAE",workload=static_workload,n_hid=64,n_dim=32,k=3)
	print("static_cost: ",kmeans_cost)

	# hill_cost, hill_partitions = hillclimb.partition(workload=workload)
'''

# Ideal situation
if __name__ == '__main__':
	dynamic_datasets, static_data = dataset.dynamic_workloads(15,10)
	dynamic_vpgae_cost = static_vpgae_cost = dynamic_vpgaeb_cost = static_vpgaeb_cost = 0
	
	for data in dynamic_datasets:
		workload=Workload(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8])
		vpgae_cost,vpgae_partitions = VPGAE.partition(algo_type="VPGAE",workload=workload,n_hid=64,n_dim=32,k=3)
		vpgaeb_cost,vpgaeb_partitions = VPGAE.partition(algo_type="VPGAE-B",workload=workload,n_hid=64,n_dim=32,k=3,origin_candidate_length=3,beam_search_width=1)

		dynamic_vpgae_cost += vpgae_cost
		dynamic_vpgaeb_cost += vpgaeb_cost
	
	static_workload=Workload(static_data[0],static_data[1],static_data[2],static_data[3],static_data[4],static_data[5],static_data[6],static_data[7],static_data[8])
	static_vpgae_cost,static_vpgae_partitions = VPGAE.partition(algo_type="VPGAE",workload=static_workload,n_hid=64,n_dim=32,k=3)
	static_vpgaeb_cost,static_vpgaeb_partitions = VPGAE.partition(algo_type="VPGAE-B",workload=static_workload,n_hid=64,n_dim=32,k=3,origin_candidate_length=3,beam_search_width=1)

	print("dynamic_vpgae_cost: ",dynamic_vpgae_cost)
	print("static_vpgae_cost: ",static_vpgae_cost)

	print("dynamic_vpgaeb_cost: ",dynamic_vpgaeb_cost)
	print("static_vpgaeb_cost: ",static_vpgaeb_cost)