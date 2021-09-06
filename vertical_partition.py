import time
import dataset
import VPGAE
import column
import row
import optimal
import hillclimb

import numpy as np

from unnecessary_data_read import fraction_of_unnecessary_data_read
from reconstruction_joins import number_of_joins
from workload_class import Workload

'''
# random dataset experiments
if __name__ == "__main__":
	attributes_num = [50]
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
			
			t1=time.time()
			beam_cost, beam_partitions = VPGAE.partition(algo_type="VPGAE-B",workload=workload,n_hid=64,n_dim=32,k=3,origin_candidate_length=3,beam_search_width=1)
			beam_time=time.time()-t1
			
			t2=time.time()
			kmeans_cost, kmeans_partitions = VPGAE.partition(algo_type="VPGAE",workload=workload,n_hid=64,n_dim=32,k=3)
			kmeans_time=time.time()-t2

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

'''
# TPC-H experiments
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

# TPC-DS experiments
if __name__ == "__main__":
	dataset_ = dataset.tpcds_workload(10)
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
	
	print("VPGAE-B costs on 24 tables:", beam_costs)
	print("VPGAE costs on 24 tables:", kmeans_costs)
	print("HILLCLIMB costs on 24 tables:", hill_costs)
	print("COLUMN costs on 24 tables:", column_costs)
	print("ROW costs on 24 tables:", row_costs)
	
	print("Unnecessary data read of VPGAE-B:", fraction_of_unnecessary_data_read(beam_partitions_list, workload_list))
	print("Unnecessary data read of VPGAE:", fraction_of_unnecessary_data_read(kmeans_partitions_list, workload_list))
	print("Unnecessary data read of HILLCLIMB:", fraction_of_unnecessary_data_read(hill_partitions_list, workload_list))
	print("Unnecessary data read of COLUMN:", fraction_of_unnecessary_data_read(column_partitions_list, workload_list))
	print("Unnecessary data read of ROW:", fraction_of_unnecessary_data_read(row_partitions_list, workload_list))

	column_RJ = np.sum(number_of_joins(column_partitions_list, workload_list))
	if column_RJ == 0:
		print("column reconstruction joins = 0")
	else:
		print("normalized reconstruction joins of VPGAE-B:", np.sum(number_of_joins(beam_partitions_list, workload_list))/column_RJ)
		print("normalized reconstruction joins of VPGAE:", np.sum(number_of_joins(kmeans_partitions_list, workload_list))/column_RJ)
		print("normalized reconstruction joins of HILLCLIMB:", np.sum(number_of_joins(hill_partitions_list, workload_list))/column_RJ)
		print("normalized reconstruction joins of COLUMN:", column_RJ/column_RJ)
		print("normalized reconstruction joins of ROW:", np.sum(number_of_joins(row_partitions_list, workload_list))/column_RJ)

	print("--------------------")


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