import time
import dataset
import VPGAE
import column
import row
import optimal
import my_cost_model
import hillclimb

import numpy as np

from unnecessary_data_read import fraction_of_unnecessary_data_read
from reconstruction_joins import number_of_joins
from workload_class import Workload

# random dataset experiments
if __name__ == "__main__":
	attributes_num = [75]
	for a_num in attributes_num:
		print("tables have {} attributes.".format(a_num))
		w_num = 100
		dataset_ = dataset.random_generator(num = w_num, a_num_range = [a_num,a_num])
		
		beam_costs = []
		kmeans_costs = []
		hill_costs = []
		column_costs = []
		
		beam_times = []
		kmeans_times = []
		hill_times = []
		
		for i,data in enumerate(dataset_):
			workload = Workload(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8])
			
			t2=time.time()
			kmeans_cost, kmeans_partitions = VPGAE.partition(algo_type="VPGAE",workload=workload,n_hid=64,n_dim=32,k=3)
			kmeans_time=time.time()-t2

			t1=time.time()
			beam_cost, beam_partitions = VPGAE.partition(algo_type="VPGAE-B",workload=workload,n_hid=64,n_dim=32,k=3,origin_candidate_length=3,beam_search_width=1)
			beam_time=time.time()-t1

			t3=time.time()
			hill_cost, hill_partitions = hillclimb.partition(workload=workload)
			hill_time=time.time()-t3

			column_cost, column_partitions = column.partition(workload=workload)

			beam_costs.append(beam_cost)
			kmeans_costs.append(kmeans_cost)
			hill_costs.append(hill_cost)
			column_costs.append(column_cost)

			beam_times.append(beam_time)
			kmeans_times.append(kmeans_time)
			hill_times.append(hill_time)

		print("Avg. VPGAE-B cost:{}".format(np.mean(beam_costs)))
		print("Avg. VPGAE cost:{}".format(np.mean(kmeans_costs)))
		print("Avg. HILLCLIMB cost:{}".format(np.mean(hill_costs)))
		print("Avg. COLUMN cost:{}".format(np.mean(column_costs)))
		
		print("Avg. VPGAE-B time:{}".format(np.mean(beam_times)))
		print("Avg. VPGAE time:{}".format(np.mean(kmeans_times)))
		print("Avg. HILLCLIMB time:{}".format(np.mean(hill_times)))

		print("--------------------")


'''
# TPC-H benchmark experiments
if __name__ == "__main__":
	dataset_ = dataset.tpch_workload(10)
	beam_costs = []
	kmeans_costs = []
	hill_costs = []
	column_costs = []
	row_costs = []

	beam_partitions_list = []
	kmeans_partitions_list = []
	hill_partitions_list = []
	column_partitions_list = []
	row_partitions_list = []
	workload_list = []

	for i,data in enumerate(dataset_):
		workload = Workload(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8])
		
		beam_cost, beam_partitions = VPGAE.partition(algo_type="VPGAE-B",workload=workload,n_hid=4,n_dim=16,k=3,origin_candidate_length=3,beam_search_width=3)
		kmeans_cost, kmeans_partitions = VPGAE.partition(algo_type="VPGAE",workload=workload,n_hid=4,n_dim=16,k=3)
		
		hill_cost, hill_partitions = hillclimb.partition(workload=workload)
		column_cost, column_partitions = column.partition(workload=workload)
		row_cost, row_partitions = row.partition(workload=workload)

		beam_costs.append(beam_cost)
		kmeans_costs.append(kmeans_cost)
		hill_costs.append(hill_cost)
		column_costs.append(column_cost)
		row_costs.append(row_cost)

		beam_partitions_list.append(beam_partitions)
		kmeans_partitions_list.append(kmeans_partitions)
		hill_partitions_list.append(hill_partitions)
		column_partitions_list.append(column_partitions)
		row_partitions_list.append(row_partitions)
		workload_list.append(workload)
	
	print("VPGAE-B costs on 8 tables:", beam_costs)
	print("VPGAE costs on 8 tables:", kmeans_costs)
	print("HILLCLIMB costs on 8 tables:", hill_costs)
	print("COLUMN costs on 8 tables:", column_costs)
	print("ROW costs on 8 tables:", row_costs)
	
	print("Unnecessary data read of VPGAE-B:", fraction_of_unnecessary_data_read(beam_partitions_list, workload_list))
	print("Unnecessary data read of VPGAE:", fraction_of_unnecessary_data_read(kmeans_partitions_list, workload_list))
	print("Unnecessary data read of HILLCLIMB:", fraction_of_unnecessary_data_read(hill_partitions_list, workload_list))
	print("Unnecessary data read of COLUMN:", fraction_of_unnecessary_data_read(column_partitions_list, workload_list))
	print("Unnecessary data read of ROW:", fraction_of_unnecessary_data_read(row_partitions_list, workload_list))

	column_RJ = np.sum(number_of_joins(column_partitions_list, workload_list))
	print("normalized reconstruction joins of VPGAE-B:", np.sum(number_of_joins(beam_partitions_list, workload_list))/column_RJ)
	print("normalized reconstruction joins of VPGAE:", np.sum(number_of_joins(kmeans_partitions_list, workload_list))/column_RJ)
	print("normalized reconstruction joins of HILLCLIMB:", np.sum(number_of_joins(hill_partitions_list, workload_list))/column_RJ)
	print("normalized reconstruction joins of COLUMN:", column_RJ/column_RJ)
	print("normalized reconstruction joins of ROW:", np.sum(number_of_joins(row_partitions_list, workload_list))/column_RJ)

	print("--------------------")
'''

'''
# TPC-DS benchmark experiments
if __name__ == "__main__":
	dataset_ = dataset.tpcds_workload()
	beam_costs = []
	kmeans_costs = []
	hill_costs = []
	column_costs = []
	row_costs = []
	hyrise_costs = []
	navathe_costs = []
	o2p_costs = []
	
	beam_partitions_list = []
	kmeans_partitions_list = []
	hill_partitions_list = []
	column_partitions_list = []
	row_partitions_list = []
	workload_list = []
	hyrise_partitions_list = []
	navathe_partitions_list = []
	o2p_partitions_list = []

	for i,data in enumerate(dataset_):
		workload = Workload(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8])
		
		beam_cost, beam_partitions = VPGAE.partition(algo_type="VPGAE-B",workload=workload,n_hid=16,n_dim=32,k=3,origin_candidate_length=3,beam_search_width=3)
		kmeans_cost, kmeans_partitions = VPGAE.partition(algo_type="VPGAE",workload=workload,n_hid=16,n_dim=32,k=3)
		
		hill_cost, hill_partitions = hillclimb.partition(workload=workload)
		column_cost, column_partitions = column.partition(workload=workload)
		row_cost, row_partitions = row.partition(workload=workload)

		beam_costs.append(beam_cost)
		kmeans_costs.append(kmeans_cost)
		hill_costs.append(hill_cost)
		column_costs.append(column_cost)
		row_costs.append(row_cost)

		beam_partitions_list.append(beam_partitions)
		kmeans_partitions_list.append(kmeans_partitions)
		hill_partitions_list.append(hill_partitions)
		column_partitions_list.append(column_partitions)
		row_partitions_list.append(row_partitions)
		workload_list.append(workload)
	
	hyrise_partitions_list = [[[11], [10], [7], [13, 8, 6, 5, 4, 3, 2], [12], [9, 1]],
						[[9, 8, 7, 6, 5], [2], [4, 3], [1]],
						[[3, 1], [10], [15], [8], [9], [7], [11], [4], [28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 14, 13, 12, 6, 5, 2]],
						[[3, 1], [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 2]],
						[[3, 1], [6, 5, 4, 2]],
						[[5, 4, 3, 1], [10, 9, 8, 7, 6, 2]],
						[[3, 1], [2]],
						[[3, 2, 1]],
						[[15, 14], [22], [12], [13], [2], [1], [11], [20, 19, 18, 17, 16, 10, 7, 6, 5, 4, 3], [21], [9, 8]],
						[[29, 27, 22, 21, 20, 19, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 5, 4, 3], [25], [24], [26], [23, 7], [18], [28], [2], [6, 1]],
						[[12, 2], [7, 1], [31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 11, 10, 9, 8, 6, 5, 4, 3]],
						[[10, 9], [2], [4, 3], [5], [11, 8], [1], [18, 17, 16, 15, 14, 13, 12, 7, 6]],
						[[5, 1], [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 4, 3, 2]],
						[[5], [20, 19, 18, 17, 16, 15, 14, 13, 12, 8, 7, 6, 4, 2, 1], [11, 10, 9, 3]],
						[[1], [2], [4], [3], [5]],
						[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
						[[15, 10, 1], [19, 18, 17, 16, 14, 13, 12, 11, 9, 8, 7, 6, 5, 4, 3, 2]],
						[[1, 2, 3, 4, 5, 6, 7, 8, 9]],
						[[4, 2, 1], [3]],
						[[27, 12, 8, 1], [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 11, 10, 9, 7, 6, 5, 4, 3, 2]],
						[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]],
						[[16, 15, 14, 3, 1], [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 2]],
						[[28, 21, 19, 17, 16, 5], [4], [15, 14, 12, 3], [34, 33, 32, 31, 30, 29, 27, 26, 25, 24, 23, 20, 18, 13, 11, 10, 9, 8, 7, 6, 2], [22, 1]],
						[[20], [9], [4], [13], [3], [14], [1], [8], [23, 7], [2], [22, 21, 19, 18, 15, 12], [17], [16], [11, 5], [10, 6]]
						]

	for i in range(len(workload_list)):
		hyrise_costs.append(my_cost_model.calculate_cost_fair(hyrise_partitions_list[i],workload_list[i]))

	navathe_partitions_list = 	[[[13, 12], [11], [9], [10], [7], [2], [3], [4], [5], [6], [8], [1]],
							[[9, 4, 3, 2], [5], [6], [7], [8], [1]],
							[[28, 8], [10], [9], [7, 3], [4], [11], [15], [2], [5], [6], [12], [13], [14], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [1]],
							[[14], [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
							[[6], [5, 4, 3, 2, 1]],
							[[10], [5, 4, 3], [2], [6], [7], [8], [9], [1]],
							[[3, 1], [2]],
							[[3, 2, 1]],
							[[20, 12], [15, 14], [8], [21], [9], [13], [11], [22], [2], [3], [4], [5], [6], [7], [10], [16], [17], [18], [19], [1]],
							[[29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
							[[31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
							[[18, 3], [11, 8], [10, 9], [5], [4], [2], [6], [7], [12], [13], [14], [15], [16], [17], [1]],
							[[26], [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
							[[20], [2], [11, 10, 9, 3], [4], [5], [6], [7], [8], [12], [13], [14], [15], [16], [17], [18], [19], [1]],
							[[3, 2], [5, 4, 1]],
							[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
							[[19], [15, 10], [2], [3], [4], [5], [6], [7], [8], [9], [11], [12], [13], [14], [16], [17], [18], [1]],
							[[1, 2, 3, 4, 5, 6, 7, 8, 9]],
							[[3], [4, 2, 1]],
							[[26], [27, 12, 8], [2], [3], [4], [5], [6], [7], [9], [10], [11], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [1]],
							[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]],
							[[34], [16, 15, 14, 3], [2], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [1]],
							[[34, 4], [28, 21, 19, 17, 16, 5], [22], [15, 14, 12, 3], [2], [6], [7], [8], [9], [10], [11], [13], [18], [20], [23], [24], [25], [26], [27], [29], [30], [31], [32], [33], [1]],
							[[22, 9], [20, 13], [16], [23, 7], [11, 5], [14], [3], [8], [4], [10, 6], [17], [2], [12], [15], [18], [19], [21], [1]]
							]

	for i in range(len(workload_list)):
		navathe_costs.append(my_cost_model.calculate_cost_fair(navathe_partitions_list[i],workload_list[i]))

	o2p_partitions_list = 	[[[13, 12], [11], [9], [10], [7], [2], [3], [8, 6, 5, 4], [1]],
						[[9, 4, 3, 2], [8, 7, 6, 5], [1]],
						[[28, 8], [10], [9], [7], [3], [4], [11], [15, 2], [5], [6], [12], [13], [14], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [1]],
						[[14], [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
						[[6], [5, 4, 3, 2, 1]],
						[[10], [3], [4], [5], [9, 8, 7, 6, 2], [1]],
						[[3, 1], [2]],
						[[3],[2,1]],
						[[20, 12], [14], [15], [8], [21], [9], [13], [11], [22], [2], [3], [4], [5], [6], [7], [10], [16], [17], [18], [19], [1]],
						[[29, 7], [28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 6, 5, 4, 3, 2, 1]],
						[[31, 12], [30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
						[[18, 3], [11, 8], [10, 9], [5], [4], [2], [6], [7], [12], [13], [14], [15], [16], [17], [1]],
						[[26], [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
						[[20, 2], [11, 10, 9, 3], [5, 4], [19, 18, 17, 16, 15, 14, 13, 12, 8, 7, 6, 1]],
						[[3, 2], [5, 4, 1]],
						[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
						[[19], [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
						[[1, 2, 3, 4, 5, 6, 7, 8, 9]],
						[[4, 2, 1], [3]],
						[[26], [27, 12, 8], [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 11, 10, 9, 7, 6, 5, 4, 3, 2], [1]],
						[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]],
						[[34], [3], [14], [15], [16], [2], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [17], [18], [19], [20], [21], [22], [23], [24], [25], [33, 32, 31, 30, 29, 28, 27, 26], [1]],
						[[34, 4], [28, 21, 19, 17, 16, 5], [22], [3], [12], [14], [15], [2], [6], [7], [8], [9], [10], [11], [13], [18], [20], [23], [24], [25], [26], [27], [29], [30], [31], [32], [33], [1]],
						[[22, 9], [13], [20], [16], [7], [23], [5], [14, 11], [3], [8], [4], [6], [10], [17], [2], [12], [15], [18], [19], [21], [1]]
						]
	
	for i in range(len(workload_list)):
		o2p_costs.append(my_cost_model.calculate_cost_fair(o2p_partitions_list[i],workload_list[i]))

	print("VPGAE-B costs on 24 tables:", beam_costs)
	print("VPGAE costs on 24 tables:", kmeans_costs)
	print("HILLCLIMB costs on 24 tables:", hill_costs)
	print("COLUMN costs on 24 tables:", column_costs)
	print("ROW costs on 24 tables:", row_costs)

	print("HYRISE costs on 24 tables:", hyrise_costs)
	print("NAVATHE costs on 24 tables:", navathe_costs)
	print("O2P costs on 24 tables:", o2p_costs)
	
	print("Unnecessary data read of VPGAE-B:", fraction_of_unnecessary_data_read(beam_partitions_list, workload_list))
	print("Unnecessary data read of VPGAE:", fraction_of_unnecessary_data_read(kmeans_partitions_list, workload_list))
	print("Unnecessary data read of HILLCLIMB:", fraction_of_unnecessary_data_read(hill_partitions_list, workload_list))
	print("Unnecessary data read of COLUMN:", fraction_of_unnecessary_data_read(column_partitions_list, workload_list))
	print("Unnecessary data read of ROW:", fraction_of_unnecessary_data_read(row_partitions_list, workload_list))

	print("Unnecessary data read of HYRISE:", fraction_of_unnecessary_data_read(hyrise_partitions_list, workload_list))
	print("Unnecessary data read of NAVATHE:", fraction_of_unnecessary_data_read(navathe_partitions_list, workload_list))
	print("Unnecessary data read of O2P:", fraction_of_unnecessary_data_read(o2p_partitions_list, workload_list))

	column_RJ = np.sum(number_of_joins(column_partitions_list, workload_list))
	if column_RJ == 0:
		print("column reconstruction joins = 0")
	else:
		print("normalized reconstruction joins of VPGAE-B:", np.sum(number_of_joins(beam_partitions_list, workload_list))/column_RJ)
		print("normalized reconstruction joins of VPGAE:", np.sum(number_of_joins(kmeans_partitions_list, workload_list))/column_RJ)
		print("normalized reconstruction joins of HILLCLIMB:", np.sum(number_of_joins(hill_partitions_list, workload_list))/column_RJ)
		print("normalized reconstruction joins of COLUMN:", column_RJ/column_RJ)
		print("normalized reconstruction joins of ROW:", np.sum(number_of_joins(row_partitions_list, workload_list))/column_RJ)

		print("normalized reconstruction joins of HYRISE:", np.sum(number_of_joins(hyrise_partitions_list, workload_list))/column_RJ)
		print("normalized reconstruction joins of NAVATHE:", np.sum(number_of_joins(navathe_partitions_list, workload_list))/column_RJ)
		print("normalized reconstruction joins of O2P:", np.sum(number_of_joins(o2p_partitions_list, workload_list))/column_RJ)

	print("--------------------")
'''

'''
# workload size experiments
if __name__ == "__main__":
	data = dataset.lineitem(10)
	
	print("workload size = 17")
	workload = Workload(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8])
	
	beam_cost, beam_partitions = VPGAE.partition(algo_type="VPGAE-B",workload=workload,n_hid=4,n_dim=16,k=5,origin_candidate_length=3,beam_search_width=3)
	kmeans_cost, kmeans_partitions = VPGAE.partition(algo_type="VPGAE",workload=workload,n_hid=4,n_dim=16,k=5)
	
	hill_cost, hill_partitions = hillclimb.partition(workload=workload)
	column_cost, column_partitions = column.partition(workload=workload)

	print("VPGAE-B: ",beam_cost)
	print("VPGAE: ", kmeans_cost)
	print("HILLCLIMB: ", hill_cost)
	print("COLUMN: ", column_cost)
	print("--------------------")

	for i in range(data[0]-1):
		print("workload size = {}".format(16-i))
		data[3][16-i] = 0
		workload = Workload(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8])
		
		kmeans_cost, kmeans_partitions = VPGAE.partition(algo_type="VPGAE",workload=workload,n_hid=4,n_dim=16,k=5)
		beam_cost, beam_partitions = VPGAE.partition(algo_type="VPGAE-B",workload=workload,n_hid=4,n_dim=16,k=5,origin_candidate_length=3,beam_search_width=3)
		
		hill_cost, hill_partitions = hillclimb.partition(workload=workload)
		column_cost, column_partitions = column.partition(workload=workload)

		print("VPGAE-B: ",beam_cost)
		print("VPGAE: ", kmeans_cost)
		print("HILLCLIMB: ", hill_cost)
		print("COLUMN: ", column_cost)
		print("--------------------")
'''

'''
# HAP benchmark experiments
if __name__ == "__main__":
	queries_number_list = [10,15,20]
	dataset_ = dataset.HAP(queries_number_list)
	for i,data in enumerate(dataset_):
		print("Workload has {} queries.".format(queries_number_list[i]))
		table_name = ["HAP"]

		workload = Workload(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8])
		
		t1=time.time()
		beam_cost, beam_partitions = VPGAE.partition(algo_type="VPGAE-B",workload=workload,n_hid=4,n_dim=16,k=5,origin_candidate_length=3,beam_search_width=3)
		beam_time = time.time()-t1

		t2=time.time()
		kmeans_cost, kmeans_partitions = VPGAE.partition(algo_type="VPGAE",workload=workload,n_hid=4,n_dim=16,k=5)
		kmeans_time=time.time()-t2
		
		t3=time.time()
		hill_cost, hill_partitions = hillclimb.partition(workload=workload)
		hill_time=time.time()-t3

		column_cost, column_partitions = column.partition(workload=workload)

		print("VPGAE-B cost on HAP wide table:",beam_cost)
		print("VPGAE cost on HAP wide table:",kmeans_cost)
		print("HILLCLIMB cost on HAP wide table:",hill_cost)
		print("COLUMN costs on HAP wide table:",column_cost)

		print("VPGAE-B time:{}, VPGAE time:{}, HILLCLIMB time:{}".format(beam_time,kmeans_time,hill_time))

		print("--------------------")
'''