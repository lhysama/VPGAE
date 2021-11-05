import numpy as np
import copy

# Usage matrix to affinity matrix
def usage_matrix_to_affinity_matrix(usage_matrix):
	attribute_num = usage_matrix.shape[1] - 1
	query_num = usage_matrix.shape[0]
	affnity_matrix = np.zeros((attribute_num,attribute_num),dtype = np.float32)
	for i in range(attribute_num):
		for j in range(i,attribute_num):
			temp = 0
			for z in range(query_num):
				if usage_matrix[z][i] == 1 and usage_matrix[z][j] == 1:
					temp += usage_matrix[z][usage_matrix.shape[1] - 1]
			affnity_matrix[i][j] = temp
			affnity_matrix[j][i] = temp
	return affnity_matrix

# Define workload profile class for VPGAE and VPGAE-B
class VPGAE_Workload():
	def __init__(self,query_num, attribute_num, attribute_usage_list, frequence_list, selectivity_list=None,length_of_attributes=None, scan_key_list=None, cluster_index = None, cardinality = None):
		self.required_attributes = attribute_usage_list
		self.query_num = query_num
		self.attribute_num = attribute_num

		self.usage_matrix = np.zeros((query_num,attribute_num+1),dtype = np.float32)
		for row in range(query_num):
			for col in attribute_usage_list[row]:
				self.usage_matrix[row][col-1] = 1.0
		
		self.subsets = self.init_subsets(self.attribute_num, self.required_attributes)
		# print(sets)
		# print(self.usage_matrix)

		delete_attr = []
		for subset in self.subsets:
			if len(subset) > 1:
				delete_attr += subset[1:]

		if len(delete_attr) != 0:
			self.new_usage_matrix = np.delete(self.usage_matrix, np.array(delete_attr)-1, axis = 1)
		else:
			self.new_usage_matrix = self.usage_matrix

		self.map_newindex_2_oldindex = {}
		
		for new_index in range(self.new_usage_matrix.shape[1]-1):
			old_index_list = []
			for old_index in range(self.usage_matrix.shape[1]-1):
				if False not in (self.new_usage_matrix[:,new_index] == self.usage_matrix[:,old_index]):
					old_index_list.append(old_index+1)
				self.map_newindex_2_oldindex[new_index+1] = old_index_list

		# print(self.map_newindex_2_oldindex)
		# print(self.new_usage_matrix)

		self.freq = np.array(frequence_list, dtype = np.float32)
		self.new_usage_matrix[:,-1] = self.freq
		self.affnity_matrix = usage_matrix_to_affinity_matrix(self.new_usage_matrix)
		
		if selectivity_list != None:
			self.selectivity = np.array(selectivity_list, dtype = np.float32)
		else:
			self.selectivity = None

		if length_of_attributes != None:
			self.length_of_attributes = length_of_attributes
		else:
			self.length_of_attributes = None

		if scan_key_list != None:
			self.scan_key = scan_key_list
		else:
			self.scan_key = None	
		
		if cluster_index != None:
			self.cluster_index = cluster_index
		else:
			self.cluster_index = None
		
		if cardinality != None:
			self.cardinality = cardinality
		else:
			self.cardinality = None

	def init_subsets(self, attribute_num, required_attributes):
		partitions = [[i+1 for i in range(attribute_num)]]
		for query_attr in required_attributes:
			temp_partitions = copy.deepcopy(partitions)
			for partition in temp_partitions:
				accessed_attrs = []
				ignored_attrs = []

				for attr in query_attr:
					if attr in partition:
						accessed_attrs.append(attr)
				for attr in partition:
					if attr not in accessed_attrs:
						ignored_attrs.append(attr)

				partitions.remove(partition)
				if len(accessed_attrs) > 0:
					partitions.append(accessed_attrs)
				if len(ignored_attrs) > 0:
					partitions.append(ignored_attrs)
		return partitions

# Define workload profile class for other baselines
class Workload():
	def __init__(self,query_num, attribute_num, attribute_usage_list, frequence_list, selectivity_list=None,length_of_attributes=None, scan_key_list=None, cluster_index = None, cardinality = None):
		self.required_attributes = attribute_usage_list
		self.query_num = query_num
		self.attribute_num = attribute_num
		self.usage_matrix = np.zeros((query_num,attribute_num+1),dtype = np.float32)
		for row in range(query_num):
			for col in attribute_usage_list[row]:
				self.usage_matrix[row][col-1] = 1.0
		self.freq = np.array(frequence_list, dtype = np.float32)
		self.usage_matrix[:,-1] = self.freq
		self.affnity_matrix = usage_matrix_to_affinity_matrix(self.usage_matrix)
		
		if selectivity_list != None:
			self.selectivity = np.array(selectivity_list, dtype = np.float32)
		else:
			self.selectivity = None

		if length_of_attributes != None:
			self.length_of_attributes = length_of_attributes
		else:
			self.length_of_attributes = None

		if scan_key_list != None:
			self.scan_key = scan_key_list
		else:
			self.scan_key = None	
		
		if cluster_index != None:
			self.cluster_index = cluster_index
		else:
			self.cluster_index = None
		
		if cardinality != None:
			self.cardinality = cardinality
		else:
			self.cardinality = None