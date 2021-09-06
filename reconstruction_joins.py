import copy

def number_of_joins(partition_scheme_list, workload_list):
	list_ = []
	for i in range(len(workload_list)):
		workload = workload_list[i]
		partition_scheme = partition_scheme_list[i]
		
		if workload.query_num == 0:
			all_partition_number = 0
		else:
			required_attributes = copy.deepcopy(workload.required_attributes)
			partition_scheme = copy.deepcopy(partition_scheme)
			all_partition_number = 0

			for query_attributes in required_attributes:
				required_partitions_id = []
				for i,partition in enumerate(partition_scheme):
					for attr in query_attributes:
						if attr in partition:
							required_partitions_id.append(i)
				
				required_partitions_id = list(set(required_partitions_id))
				all_partition_number += (len(required_partitions_id)-1)

		list_.append(all_partition_number * workload.cardinality)
	
	return list_