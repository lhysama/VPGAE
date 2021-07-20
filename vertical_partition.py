import sys
import time
import math
import argparse
import torch
import copy
import random
import dataset
import my_cost_model
import visualization

import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import more_itertools as mit

from unnecessary_data_read import fraction_of_unnecessary_data_read
from reconstruction_joins import number_of_joins

from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn import metrics
from sklearn.decomposition import PCA

from torch_geometric.nn import SGConv
from torch_geometric.data import Data
from tqdm import tqdm

def reset_seed():
	seed = 777
	torch.manual_seed(seed) # CPU pytorch seed
	torch.cuda.manual_seed(seed) # GPU pytorch seed
	torch.cuda.manual_seed_all(seed)# all GPUs pytorch seed
	np.random.seed(seed)  # Numpy module.
	random.seed(seed)  # Python random module.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

# usage matrix to affinity matrix
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

# loss function
def my_loss(output,label_x):
	mse = nn.MSELoss()
	loss = mse(output,label_x)

	return loss

# optimal method
def optimal(workload):
	best_cost = float('inf')
	best_partitioning_scheme = []
	lst = [i+1 for i in range(workload.attribute_num)]

	for k in range(1,len(lst)+1):
		for partition in tqdm(mit.set_partitions(lst,k)):
			temp_cost = my_cost_model.calculate_cost_fair(partition,workload)
			if temp_cost < best_cost:
				best_cost = temp_cost
				best_partitioning_scheme = partition

	return best_cost, best_partitioning_scheme
	
def row(workload):
	partition = [[i+1 for i in range(workload.attribute_num)]]

	return my_cost_model.calculate_cost_fair(partition,workload),partition

def column(workload):
	partition = [[i+1] for i in range(workload.attribute_num)]
	
	return my_cost_model.calculate_cost_fair(partition,workload),partition

def hill_climb_optim(workload):
	candidates = [[i+1] for i in range(workload.attribute_num)]
	best_cost = my_cost_model.calculate_cost_fair(candidates,workload)
	best_partitioning_scheme = candidates
	
	candidates = [best_partitioning_scheme, best_cost]
	# start search
	while True:
		temp_best_cost = float('inf')
		temp_best_partitioning_schemes = []
		partitioning_scheme = candidates[0]
		# merge two partitions
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

		# if current best partitioning scheme is better than the history best partitioning scheme, then update best_cost, best_partitioning_scheme and candidates
		if(temp_best_cost < best_cost):
			best_cost = temp_best_cost
			best_partitioning_scheme = temp_best_partitioning_schemes
			candidates = [best_partitioning_scheme,best_cost]
		# if current best partitioning scheme is worse than the history best partitioning scheme, stop search
		else:
			break

	return best_cost,best_partitioning_scheme

def beam_search(embedding,workload,origin_candidate_length,beam_search_width):
	kmax = embedding.shape[0]
	
	best_cost = float('inf')
	best_partitioning_scheme = []

	candidates = []
	for k in range(1 , kmax+1):
		kmeans = KMeans(n_clusters = k).fit(embedding)
		labels = kmeans.labels_
		
		label_max = max(labels)
		partitioning_scheme = []
		for i in range(label_max + 1):
			partition = []
			for idx,j in enumerate(labels):
				if(i == j):
					partition.append(idx+1)
			if len(partition) == 0:
				continue
			else:
				partitioning_scheme.append(partition)
		
		total_cost = my_cost_model.calculate_cost_fair(partitioning_scheme,workload)
		
		if total_cost <= best_cost:
			best_cost = total_cost
			best_partitioning_scheme = partitioning_scheme

		candidates.append([partitioning_scheme, total_cost])
	
	candidates.sort(key = lambda tup:tup[1])
	
	candidates = candidates[:origin_candidate_length]
	
	# start beam search
	if beam_search_width > 1: # if beam_search_width is greater than 1, we need a list to record intermediate results	
		while True:
			all_candidates = []
			for candidate in candidates:
				partitioning_scheme = candidate[0]

				# merge two partitions
				for i in range(len(partitioning_scheme)-1):
					for j in range(i+1,len(partitioning_scheme)):
						temp_partitioning_scheme = copy.deepcopy(partitioning_scheme)
						temp_partitioning_scheme.remove(partitioning_scheme[i])
						temp_partitioning_scheme.remove(partitioning_scheme[j])
						temp_partitioning_scheme.append(partitioning_scheme[i]+partitioning_scheme[j])

						temp_cost = my_cost_model.calculate_cost_fair(temp_partitioning_scheme,workload)
						if temp_cost < best_cost:
							if len(all_candidates) < beam_search_width:
								all_candidates.append([temp_partitioning_scheme,temp_cost])
								all_candidates.sort(key = lambda tup:tup[1])
							else:
								if temp_cost < all_candidates[beam_search_width-1][1]:
									all_candidates.remove(all_candidates[beam_search_width-1])
									all_candidates.append([temp_partitioning_scheme,temp_cost])
									all_candidates.sort(key = lambda tup:tup[1])

				# split a partition that includes more than 2 attributes
				for i in range(len(partitioning_scheme)):
					if len(partitioning_scheme[i]) > 1:
						for j in range(1,len(partitioning_scheme[i])):
							temp_partitioning_scheme = copy.deepcopy(partitioning_scheme)
							temp_partitioning_scheme.remove(partitioning_scheme[i])
							temp_partitioning_scheme.append(partitioning_scheme[i][0:j])
							temp_partitioning_scheme.append(partitioning_scheme[i][j:len(partitioning_scheme[i])])
							temp_cost = my_cost_model.calculate_cost_fair(temp_partitioning_scheme,workload)
						
							if temp_cost < best_cost:
								if len(all_candidates) < beam_search_width:
									all_candidates.append([temp_partitioning_scheme,temp_cost])
									all_candidates.sort(key = lambda tup:tup[1])
								else:
									if temp_cost < all_candidates[beam_search_width-1][1]:
										all_candidates.remove(all_candidates[beam_search_width-1])
										all_candidates.append([temp_partitioning_scheme,temp_cost])
										all_candidates.sort(key = lambda tup:tup[1])

			if len(all_candidates) == 0:
				break
			
			if all_candidates[0][1] < best_cost:
				best_cost = all_candidates[0][1]
				best_partitioning_scheme = all_candidates[0][0]
				candidates = all_candidates
			else:
				break
				
	else: #  if beam_search_width is equal to 1, we only need update the best partitioning scheme	
		while True:
			flag = False
			for candidate in candidates:
				partitioning_scheme = candidate[0]
				# merge two partitions
				for i in range(len(partitioning_scheme)-1):
					for j in range(i+1,len(partitioning_scheme)):
						temp_partitioning_scheme = copy.deepcopy(partitioning_scheme)
						temp_partitioning_scheme.remove(partitioning_scheme[i])
						temp_partitioning_scheme.remove(partitioning_scheme[j])
						temp_partitioning_scheme.append(partitioning_scheme[i]+partitioning_scheme[j])
						
						temp_cost = my_cost_model.calculate_cost_fair(temp_partitioning_scheme,workload)
						if temp_cost < best_cost:
							best_cost = temp_cost
							best_partitioning_scheme = temp_partitioning_scheme
							flag = True

				# split a partition that includes more than 2 attributes
				for i in range(len(partitioning_scheme)):
					if len(partitioning_scheme[i]) > 1:
						for j in range(1,len(partitioning_scheme[i])):
							temp_partitioning_scheme = copy.deepcopy(partitioning_scheme)
							temp_partitioning_scheme.remove(partitioning_scheme[i])
							temp_partitioning_scheme.append(partitioning_scheme[i][0:j])
							temp_partitioning_scheme.append(partitioning_scheme[i][j:len(partitioning_scheme[i])])

							temp_cost = my_cost_model.calculate_cost_fair(temp_partitioning_scheme,workload)
							if temp_cost < best_cost:
								best_cost = temp_cost
								best_partitioning_scheme = temp_partitioning_scheme
								flag = True
			
			if flag:
				candidates = [[best_partitioning_scheme,best_cost]]
			else:
				break

	
	return best_cost,best_partitioning_scheme

def simple_kmeans_search(embedding,workload):
	kmax = embedding.shape[0]

	best_cost = float('inf')
	best_partitioning_scheme = []

	for k in range(1 , kmax+1):
		kmeans = KMeans(n_clusters = k).fit(embedding)
		labels = kmeans.labels_
		
		label_max = max(labels)
		partitioning_scheme = []
		for i in range(label_max + 1):
			partition = []
			for idx,j in enumerate(labels):
				if(i == j):
					partition.append(idx+1)
			partitioning_scheme.append(partition)

		total_cost = my_cost_model.calculate_cost_fair(partitioning_scheme,workload)
		# print(total_cost)

		if total_cost <= best_cost:
			best_cost = total_cost
			best_partitioning_scheme = partitioning_scheme

	return best_cost,best_partitioning_scheme


# graph autoencoder
class GAE(nn.Module):
	def __init__(self,n_feat,n_hid,n_dim,k):
		super(GAE,self).__init__()
		# encoder layer
		self.gc1 = SGConv(n_feat, n_dim, K = k, bias = False, cached = True, add_self_loops = False)
		
		# decoder layer
		self.layer1 = nn.Linear(n_dim, n_hid)
		self.layer2 = nn.Linear(n_hid, n_feat,bias = False)

	def forward(self,data):
		x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
		# encoder
		embedding = self.gc1(x,edge_index,edge_attr)
		
		# decoder
		x = F.relu(embedding)
		x = F.relu(self.layer1(x))
		x = self.layer2(x)

		return x,embedding

# define workload profile class
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

def main(data,n_hid,n_dim,tablename,origin_candidate_length, beam_search_width,k):
	workload = Workload(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8])
	# adjacent matrix of workload
	adj = workload.affnity_matrix

	# initialize node feature as one-hot vector
	features = torch.diag(torch.from_numpy(np.array([1.0 for i in range(adj.shape[1])],dtype=np.float32)))

	# in autoencoder, output is input
	label_x = copy.deepcopy(features)
	np.set_printoptions(threshold=1e6)

	# graph preprocess
	for i in range(adj.shape[0]):
		adj[i][i] = np.max(adj)
	adj = adj/np.max(adj)
	
	edge_index = []
	edge_attr = []
	for i in range(adj.shape[0]):
		for j in range(adj.shape[1]):
			if adj[i][j] > 0:
				edge_index.append([i,j])
				edge_attr.append(adj[i][j])

	edge_index = torch.tensor(edge_index,dtype=torch.long)
	edge_attr = torch.tensor(edge_attr,dtype=torch.float)

	data = Data(x=features, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)

	model = GAE(features.shape[0],n_hid,n_dim,k)

	optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)
	
	min_loss = float('inf')
	# early-stop
	patience = 15
	step = 0
	optimal_model = None

	t1 = time.time()
	row_list = [i for i in range(features.shape[1])]
	
	for i in range(1000):
		# if early-stop is triggered
		if(step >= patience):
			break

		# split train set and validation set
		train_index = random.sample(row_list, max(1,int((1/4)*features.shape[1])))
		valid_index = []
		for ele in row_list:
			if ele not in train_index:
				valid_index.append(ele)

		model.train()
		optimizer.zero_grad()
		output, embedding = model(data)

		train_loss = my_loss(output[train_index],label_x[train_index])
		train_loss.backward()
		optimizer.step()

		model.eval()
		with torch.no_grad():
			output, embedding = model(data)
		valid_loss = my_loss(output[valid_index], label_x[valid_index])

		if(valid_loss.item() < min_loss):
			min_loss = valid_loss.item()
			step = 0
			optimal_model = copy.deepcopy(model)
		else:
			step += 1

	model = optimal_model
	model.eval()
	with torch.no_grad():
		output,embedding = model(data)
	embedding = embedding.numpy()

	gae_time = time.time() - t1

	t2 = time.time()
	beam_cost,beam_partitions = beam_search(embedding, workload, origin_candidate_length, beam_search_width)
	t3 = time.time()

	kmeans_cost,kmeans_partitions = simple_kmeans_search(embedding, workload)
	t4 = time.time()

	hill_cost,hill_partitions = hill_climb_optim(workload)
	t5 = time.time()

	# optimal_cost,optimal_partitions = optimal(workload)

	column_cost,column_partitions = column(workload)
	# print(column_cost)
	row_cost, row_partitions = row(workload)
	
	# return beam_cost,kmeans_cost,hill_cost,column_cost,t3-t2+gae_time,t4-t3+gae_time,t5-t4
	return beam_cost,kmeans_cost,hill_cost,beam_partitions,workload

'''
# random dataset experients
if __name__ == "__main__":
	attributes_num = [150]
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
			reset_seed()
			beam_cost,kmeans_cost,hill_cost,column_cost,beam_time,kmeans_time,hill_time = main(data,64,32,"random dataset",3,1,3)
			
			beam_costs.append(beam_cost)
			kmeans_costs.append(kmeans_cost)
			hill_costs.append(hill_cost)
			column_costs.append(column_cost)

			beam_times.append(beam_time)
			kmeans_times.append(kmeans_time)
			hill_times.append(hill_time)
		print("avg beam_costs:{}".format(np.mean(beam_costs)))
		print("avg kmeans_costs:{}".format(np.mean(kmeans_costs)))
		print("avg hill_costs:{}".format(np.mean(hill_costs)))
		print("avg column_costs:{}".format(np.mean(column_costs)))
		
		print("avg beam_times:{}".format(np.mean(beam_times)))
		print("avg kmeans_times:{}".format(np.mean(kmeans_times)))
		print("avg hill_times:{}".format(np.mean(hill_times)))

		print("--------------------")
'''


# TPC-H experients
if __name__ == "__main__":
	dataset_ = dataset.tpch_workload(10)
	table_name = ["customer","lineitem","orders","supplier","part","partsupp","nation","region"]
	beam_costs = []
	kmeans_costs = []
	hill_costs = []
	partitioning_scheme_list = []
	workload_list = []

	for i,data in enumerate(dataset_):
		reset_seed()
		beam_cost, kmeans_cost, hill_cost, ps, workload  = main(data,4,16,table_name[i],3,3,3)
		
		beam_costs.append(beam_cost)
		kmeans_costs.append(kmeans_cost)
		hill_costs.append(hill_cost)
		partitioning_scheme_list.append(ps)
		workload_list.append(workload)
	
	print(beam_costs)
	print(kmeans_costs)
	print(hill_costs)
	# print(partitioning_scheme_list)

	print(fraction_of_unnecessary_data_read(partitioning_scheme_list, workload_list))

	join_number_list = number_of_joins(partitioning_scheme_list, workload_list)
	print(join_number_list)
	print(np.sum(join_number_list))

	print("--------------------")


'''
# workload size experients
if __name__ == "__main__":
	data = dataset.lineitem(10)
	reset_seed()
	
	print("workload size = 17")
	beam_cost,kmeans_cost,hill_cost,column_cost,beam_time,kmeans_time,hill_time = main(data,4,16,"lineitem",3,3,5)
	print("VPGAE-B: ",beam_cost)
	print("VPGAE: ", kmeans_cost)
	print("HILLCLIMB: ", hill_cost)
	print("--------------------")

	for i in range(data[0]-1):
		print("workload size = {}".format(16-i))
		reset_seed()
		data[3][16-i] = 0
		# print(data[3])
		beam_cost,kmeans_cost,hill_cost,column_cost,beam_time,kmeans_time,hill_time = main(data,4,16,"lineitem",3,3,5)
		
		print("VPGAE-B: ",beam_cost)
		print("VPGAE: ", kmeans_cost)
		print("HILLCLIMB: ", hill_cost)
		
		print("--------------------")
'''

'''
# HAP benchmark
if __name__ == "__main__":
	data = dataset.HAP()
	table_name = ["HAP"]
	beam_costs = []
	kmeans_costs = []
	hill_costs = []
	
	reset_seed()
	beam_cost,kmeans_cost,hill_cost,column_cost,beam_time,kmeans_time,hill_time = main(data,32,64,table_name[0],3,1,3)
		
	beam_costs.append(beam_cost)
	kmeans_costs.append(kmeans_cost)
	hill_costs.append(hill_cost)
		
	print(beam_costs)
	print(kmeans_costs)
	print(hill_costs)

	print(beam_time,kmeans_time,hill_time)

	print("--------------------")
'''