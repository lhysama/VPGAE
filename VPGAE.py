import sys
import time
import math
import argparse
import torch
import row
import copy
import random
import dataset
import my_cost_model
import visualization

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA

from torch_geometric.nn import SGConv
from torch_geometric.data import Data

# loss function
def my_loss(output,label_x):
	mse = nn.MSELoss()
	loss = mse(output,label_x)

	return loss

def beam_search(embedding,workload,origin_candidate_length,beam_search_width):
	kmax = embedding.shape[0]

	best_cost = float('inf')
	best_partitioning_scheme = []

	candidates = []
	for k in range(1 , kmax+1):
		if k != kmax:
			print(k)
			kmeans = KMeans(n_clusters = k).fit(embedding)
			labels = kmeans.labels_
		else:
			# fix bug when k = kmax 
			labels = [j for j in range(kmax)]

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
	
	return best_cost,best_partitioning_scheme

def simple_kmeans_search(embedding,workload):
	kmax = embedding.shape[0]

	best_cost = float('inf')
	best_partitioning_scheme = []

	for k in range(1 , kmax+1):
		if k != kmax:
			print(k)
			kmeans = KMeans(n_clusters = k).fit(embedding)
			labels = kmeans.labels_
		else:
			# fix bug when k = kmax 
			labels = [j for j in range(kmax)]
		
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

		return x, embedding

def partition(algo_type,workload,n_hid,n_dim,k,origin_candidate_length=None, beam_search_width=None):
	torch.manual_seed(seed = 777) # CPU pytorch seed
	random.seed(777)

	if workload.query_num == 0:
		return 0, row.partition(workload)

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

	if algo_type == "VPGAE-B":
		beam_cost,beam_partitions = beam_search(embedding, workload, origin_candidate_length, beam_search_width)
		return beam_cost,beam_partitions
	
	elif algo_type == "VPGAE":
		kmeans_cost,kmeans_partitions = simple_kmeans_search(embedding, workload)
		return kmeans_cost,kmeans_partitions

	else:
		print("no corresponding variant.")