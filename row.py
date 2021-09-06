import my_cost_model

def partition(workload):
	partitions = [[i+1 for i in range(workload.attribute_num)]]

	return my_cost_model.calculate_cost_fair(partitions,workload),partitions