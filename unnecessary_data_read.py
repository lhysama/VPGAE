import copy

def fraction_of_unnecessary_data_read(partitioning_scheme_list, dataset):
	all_data_need = all_data_read = 0

	# iterate workload
	for i in range(len(dataset)):
		partition_scheme = partitioning_scheme_list[i]
		workload = dataset[i]

		# if a workload includes no query, just skip it.
		if workload.query_num == 0:
			continue

		required_attributes = copy.deepcopy(workload.required_attributes)
		partition_scheme = copy.deepcopy(partition_scheme)
		data_need = data_read = 0

		# iterate query
		for query_attributes in required_attributes:
			# data need
			for attr in query_attributes:
				data_need += workload.length_of_attributes[attr-1]
			
			# find partitions referenced by query
			required_partitions_id = []
			for i,partition in enumerate(partition_scheme):
				for attr in query_attributes:
					if attr in partition:
						required_partitions_id.append(i)

			required_partitions_id = list(set(required_partitions_id))
			required_partitions = []
			for id_ in required_partitions_id:
				required_partitions.append(partition_scheme[id_])
			# data read
			for partition in required_partitions:
				for attr in partition:
					data_read += workload.length_of_attributes[attr-1]
		
		# print(data_read*workload.cardinality)

		all_data_need += data_need * workload.cardinality
		all_data_read += data_read * workload.cardinality

	# print(all_data_read)
	if all_data_read == 0:
		return None
	else:
		return 1-(all_data_need/all_data_read)