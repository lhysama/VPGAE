import my_cost_model
import copy

def partition(workload):
	candidates = [[i+1] for i in range(workload.attribute_num)]
	best_cost = my_cost_model.calculate_cost_fair(candidates,workload)
	best_partitioning_scheme = candidates
	
	candidates = [best_partitioning_scheme, best_cost]
	# Start greedy search
	while True:
		temp_best_cost = float('inf')
		temp_best_partitioning_schemes = []
		partitioning_scheme = candidates[0]
		# Merge two partitions
		for i in range(len(partitioning_scheme)-1):
			for j in range(i+1,len(partitioning_scheme)):
				temp_partitioning_scheme = copy.deepcopy(partitioning_scheme)
				temp_partitioning_scheme.remove(partitioning_scheme[i])
				temp_partitioning_scheme.remove(partitioning_scheme[j])
				temp_partitioning_scheme.append(partitioning_scheme[i]+partitioning_scheme[j])
				cost = my_cost_model.calculate_cost_fair(temp_partitioning_scheme,workload)
				if cost < temp_best_cost:
					temp_best_cost = cost
					temp_best_partitioning_schemes = temp_partitioning_scheme

		# If current best partitioning scheme is better than the history best partitioning scheme, then update best_cost, best_partitioning_scheme and candidates
		if(temp_best_cost < best_cost):
			best_cost = temp_best_cost
			best_partitioning_scheme = temp_best_partitioning_schemes
			candidates = [best_partitioning_scheme,best_cost]
		# If current best partitioning scheme is worse than the history best partitioning scheme, stop search
		else:
			break

	return best_cost, best_partitioning_scheme