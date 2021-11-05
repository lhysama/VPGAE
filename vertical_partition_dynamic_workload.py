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

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import numpy as np

from workload_class import VPGAE_Workload

def merge_queries2workload(current_query_list):
	data=[]
	data.append(len(current_query_list))
	data.append(current_query_list[0][1])
	data.append([])
	data.append([])
	data.append([])
	data.append([])
	data.append([])

	for query in current_query_list:
		data[2] += query[2]
		data[3] += query[3]
		data[4] += query[4]
		data[6] += query[6]

	data[5] = current_query_list[0][5]
	data.append(current_query_list[0][7])
	data.append(current_query_list[0][8])

	return VPGAE_Workload(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8])

# Our dynamic strategy
def vpgae_dynamic_update():
	random.seed(777)
	query_datasets = dataset.dynamic_date_dim()

	current_query_list = []
	finished_query_list = []
	time_interval_list = [random.random()*3 + 10e-8 for _ in range(len(query_datasets))]
	query_arrive_time_list = []
	query_finish_time_list = []

	for i in range(len(time_interval_list)):
		temp=0
		for j in range(0, i+1):
			temp += time_interval_list[j]
		query_arrive_time_list.append(temp)

	for i in range(len(query_arrive_time_list)):
		query_finish_time_list.append(query_arrive_time_list[i] + (2 + random.random()*9))

	print(query_finish_time_list)
	st=time.time()
	old_time = 0.0
	finished_query_num = 0
	# Pressume that the table is Row layout at the beginning
	partitioning_scheme = [[i+1 for i in range(query_datasets[0][1])]]
	time_list = []
	cost_list = []
	number_update = 0

	while(True):
		current_time = time.time()-st
		new_queries_are_submitted = False
		# Add new queries to workload
		for i in range(len(query_arrive_time_list)):
			if old_time < query_arrive_time_list[i] and query_arrive_time_list[i] <= current_time:
				current_query_list.append(query_datasets[i])
				new_queries_are_submitted = True
				# print("query {} arrived.".format(i))
		# Collect finished queries
		for i in range(len(query_finish_time_list)):
			if old_time < query_finish_time_list[i] and query_finish_time_list[i] <= current_time:
				finished_query_list.append(query_datasets[i])
				# print("query {} finished.".format(i))
				finished_query_num += 1

		# All queries are finished, break while
		if finished_query_num == len(query_datasets):
			break

		if len(current_query_list) != 0:
			current_workload = merge_queries2workload(current_query_list)
			# A new query is submitted.
			if new_queries_are_submitted:
				current_cost = my_cost_model.calculate_cost_fair(partitioning_scheme,current_workload)
				column_cost = my_cost_model.calculate_cost_fair(column.partition(current_workload)[1],current_workload)
				# if column layout is better than current partitioning scheme, update it.
				if current_cost > column_cost:
					number_update += 1
					_,partitioning_scheme = VPGAE.partition(algo_type="VPGAE",workload=current_workload,n_hid=16,n_dim=32,k=3)
					
			# There are finished queries.
			if len(finished_query_list) >= 1:
				# Remove finished queries from workload
				for finished_query in finished_query_list:
					for query in current_query_list:
						if query == finished_query:
							current_query_list.remove(query)
				
				finished_query_list = []
					
				if len(current_query_list) != 0:
					current_workload = merge_queries2workload(current_query_list)
					current_cost = my_cost_model.calculate_cost_fair(partitioning_scheme,current_workload)
					column_cost = my_cost_model.calculate_cost_fair(column.partition(current_workload)[1],current_workload)
					# if column layout is better than current partitioning scheme, update it.
					if current_cost > column_cost:
						number_update += 1
						_,partitioning_scheme = VPGAE.partition(algo_type="VPGAE",workload=current_workload,n_hid=16,n_dim=32,k=3)
				else:
					cost_list.append(0)
					time_list.append(time.time()-st)
					old_time=current_time
					continue
				
			cost_list.append(my_cost_model.calculate_cost_fair(partitioning_scheme,current_workload))
		else:
			cost_list.append(0)
		time_list.append(time.time()-st)
		old_time=current_time
	
	return time_list, cost_list, number_update
	

# Frequent update dynamic strategy
def frequent_dynamic_update():
	random.seed(777)
	query_datasets = dataset.dynamic_date_dim()

	current_query_list = []
	finished_query_list = []
	time_interval_list = [random.random()*3 + 10e-8 for _ in range(len(query_datasets))]
	query_arrive_time_list = []
	query_finish_time_list = []

	for i in range(len(time_interval_list)):
		temp=0
		for j in range(0, i+1):
			temp += time_interval_list[j]
		query_arrive_time_list.append(temp)

	for i in range(len(query_arrive_time_list)):
		query_finish_time_list.append(query_arrive_time_list[i] + (2 + random.random()*9))

	print(query_finish_time_list)
	st=time.time()
	old_time = 0.0
	finished_query_num = 0
	# Pressume that the table is Row layout at the beginning
	partitioning_scheme = [[i+1 for i in range(query_datasets[0][1])]]
	time_list = []
	cost_list = []
	number_update = 0

	while(True):
		current_time = time.time()-st
		new_queries_are_submitted = False
		# Add new queries to workload
		for i in range(len(query_arrive_time_list)):
			if old_time < query_arrive_time_list[i] and query_arrive_time_list[i] <= current_time:
				current_query_list.append(query_datasets[i])
				new_queries_are_submitted = True
				# print("query {} arrived.".format(i))
		# Collect finished queries
		for i in range(len(query_finish_time_list)):
			if old_time < query_finish_time_list[i] and query_finish_time_list[i] <= current_time:
				finished_query_list.append(query_datasets[i])
				# print("query {} finished.".format(i))
				finished_query_num += 1
		
		# All queries are finished, break while
		if finished_query_num == len(query_datasets):
			break

		if len(current_query_list):
			current_workload = merge_queries2workload(current_query_list)
			# A new query is submitted, update ps
			if new_queries_are_submitted:
				number_update += 1
				_,partitioning_scheme = VPGAE.partition(algo_type="VPGAE",workload=current_workload,n_hid=16,n_dim=32,k=3)
			
			# A query is finished, update ps
			if len(finished_query_list) >= 1:
				# Remove finished queries from workload
				for finished_query in finished_query_list:
					for query in current_query_list:
						if query == finished_query:
							current_query_list.remove(query)
				
				finished_query_list = []
				
				if len(current_query_list) != 0:
					current_workload = merge_queries2workload(current_query_list)
					number_update += 1
					_,partitioning_scheme = VPGAE.partition(algo_type="VPGAE",workload=current_workload,n_hid=16,n_dim=32,k=3)
				else:
					cost_list.append(0)
					time_list.append(time.time()-st)
					old_time=current_time
					continue

			cost_list.append(my_cost_model.calculate_cost_fair(partitioning_scheme,current_workload))
		else:
			cost_list.append(0)

		time_list.append(time.time()-st)
		old_time=current_time
	return time_list, cost_list, number_update

if __name__ == '__main__':
	vpgae_x, vpgae_y, vpgae_number_update = vpgae_dynamic_update()
	frequent_x, frequent_y, frequent_number_update = frequent_dynamic_update()
	
	fig = plt.figure(figsize=(9, 4))
	ax = axisartist.Subplot(fig, 111)

	fig.add_axes(ax)

	#通过set_visible方法设置绘图区所有坐标轴隐藏
	ax.axis[:].set_visible(False)

	#ax.new_floating_axis代表添加新的坐标轴
	ax.axis["x"] = ax.new_floating_axis(0, 0)
	#给x坐标轴加上箭头
	ax.axis["x"].set_axisline_style("->", size = 2.0)
	ax.axis["x"].label.set_text('Time step (sec)')
	#添加y坐标轴，且加上箭头
	ax.axis["y"] = ax.new_floating_axis(1, 0)
	ax.axis["y"].set_axisline_style("->", size = 2.0)
	ax.axis["y"].label.set_text("Estimated cost of dynamic workload")
	
	#设置x、y轴上刻度显示方向
	ax.axis["x"].set_axis_direction("bottom")
	ax.axis["y"].set_axis_direction("left")

	print("frequent strategy updates {} times.".format(frequent_number_update))
	print("VPGAE strategy updates {} times.".format(vpgae_number_update))
	
	line1, = plt.plot(frequent_x, frequent_y, "grey")
	line2, = plt.plot(vpgae_x, vpgae_y, "k--")

	ax.legend(handles=[line1,line2],labels=["optimal-performance", "our-update-strategy"],loc=1,fontsize=12)

	plt.savefig("./dynamic_workload_cost.pdf")


'''
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
'''