import copy
import math
import sys

class SYS_PARAM():
	cardinality = None
	length_of_TID = 4
	page_size = 1024*4
	prefetch_blocking_factor = 10

def list_cmp(list1,list2):
	if len(list1) != len(list2):
		return False
	list1_sort = sorted(list1)
	list2_sort = sorted(list2)
	for index in range(len(list1)):
		if list1_sort[index] != list2_sort[index]:
			return False

	return True

def include(list1,list2):
	for ele in list2:
		if(ele not in list1):
			return False
	return True

def sequential_scan_cost(tuple_length):
	cost = math.ceil((SYS_PARAM.cardinality * tuple_length)/(SYS_PARAM.page_size*SYS_PARAM.prefetch_blocking_factor))
	return cost

def unclustered_index_scan_cost(selectivity):
	cost = math.ceil(SYS_PARAM.cardinality * selectivity)
	return cost

def clustered_index_scan_cost(selectivity,tuple_length):
	cost = math.ceil((SYS_PARAM.cardinality*selectivity*tuple_length)/SYS_PARAM.page_size)
	return cost

def calculate_cost_fair(partitioning_scheme,workload):
	SYS_PARAM.cardinality = workload.cardinality
	total_cost = 0

	required_attributes = copy.deepcopy(workload.required_attributes)
	partitioning_scheme = copy.deepcopy(partitioning_scheme)
	
	primary_partition = None
	for partition in partitioning_scheme:
		# find primary partition
		if include(partition,workload.cluster_index):
			primary_partition = partition
			partitioning_scheme.remove(partition)
			break
	
	# iterate query
	for idx,query_attributes in enumerate(required_attributes):
		selectivity = 1.0
		# if exists primary_partition
		if primary_partition != None:
			query_use_primary_partition = False
			for attr in query_attributes:
				if attr in primary_partition:
					query_use_primary_partition = True
			if query_use_primary_partition:
				if workload.scan_key[idx] !=None and list_cmp(workload.scan_key[idx],workload.cluster_index):
					selectivity = workload.selectivity[idx]

				tuple_length = SYS_PARAM.length_of_TID
				for attr in primary_partition:
					tuple_length += workload.length_of_attributes[attr-1]
				total_cost += min(sequential_scan_cost(tuple_length),unclustered_index_scan_cost(selectivity),clustered_index_scan_cost(selectivity,tuple_length))*workload.freq[idx]
				for attr in primary_partition:
					if attr in query_attributes:
						query_attributes.remove(attr)
			
		required_partition_ids = []
		for i,partition in enumerate(partitioning_scheme):
			for attr in query_attributes:
				if attr in partition:
					required_partition_ids.append(i)
		required_partition_ids = list(set(required_partition_ids))
		required_partitions = []
		for id in required_partition_ids:
			required_partitions.append(partitioning_scheme[id])

		for partition in required_partitions:
			tuple_length = SYS_PARAM.length_of_TID
			for attr in partition:
				tuple_length += workload.length_of_attributes[attr-1]
			total_cost += min(sequential_scan_cost(tuple_length),unclustered_index_scan_cost(selectivity))*workload.freq[idx]
			
	return total_cost