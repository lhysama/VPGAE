import numpy as np

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

# Define workload profile class
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