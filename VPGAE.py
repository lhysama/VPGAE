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

def restore_ps(partitioning_scheme, workload):
	origin_partitioning_scheme = []
	for partition in partitioning_scheme:
		temp_partition = []
		for attr in partition:
			temp_partition += workload.map_newindex_2_oldindex[attr]
		origin_partitioning_scheme.append(temp_partition)
	return origin_partitioning_scheme

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

		origin_partitioning_scheme = restore_ps(partitioning_scheme, workload)
		cost = my_cost_model.calculate_cost_fair(origin_partitioning_scheme,workload)
		
		if cost <= best_cost:
			best_cost = cost
			best_partitioning_scheme = restore_ps(partitioning_scheme, workload)

		candidates.append([partitioning_scheme, cost])
	
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
					
					origin_partitioning_scheme = restore_ps(temp_partitioning_scheme, workload)
					temp_cost = my_cost_model.calculate_cost_fair(origin_partitioning_scheme,workload)
					
					if temp_cost < best_cost:
						all_candidates.append([temp_partitioning_scheme,temp_cost])
			
			# split a partition that includes more than 2 attributes
			for i in range(len(partitioning_scheme)):
				if len(partitioning_scheme[i]) > 1:
					for j in range(1,len(partitioning_scheme[i])):
						temp_partitioning_scheme = copy.deepcopy(partitioning_scheme)
						temp_partitioning_scheme.remove(partitioning_scheme[i])
						temp_partitioning_scheme.append(partitioning_scheme[i][0:j])
						temp_partitioning_scheme.append(partitioning_scheme[i][j:len(partitioning_scheme[i])])

						origin_partitioning_scheme = restore_ps(temp_partitioning_scheme, workload)
						temp_cost = my_cost_model.calculate_cost_fair(origin_partitioning_scheme,workload)
					
						if temp_cost < best_cost:
							all_candidates.append([temp_partitioning_scheme,temp_cost])

		if len(all_candidates) == 0:
			break
		
		all_candidates.sort(key = lambda tup:tup[1])		
		best_cost = all_candidates[0][1]
		best_partitioning_scheme = restore_ps(all_candidates[0][0], workload)
		candidates = all_candidates[:beam_search_width]
		
	return best_cost, best_partitioning_scheme

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
			if len(partition) == 0:
				continue
			else:
				partitioning_scheme.append(partition)

		origin_partitioning_scheme = restore_ps(partitioning_scheme, workload)
		cost = my_cost_model.calculate_cost_fair(origin_partitioning_scheme, workload)

		if cost <= best_cost:
			best_cost = cost
			best_partitioning_scheme = partitioning_scheme
	
	best_partitioning_scheme = restore_ps(best_partitioning_scheme, workload)

	return best_cost,best_partitioning_scheme

# graph autoencoder
class GAE(nn.Module):
	def __init__(self, n_feat, n_hid, n_dim, k):
		super(GAE,self).__init__()
		# encoder layer
		self.gc1 = SGConv(n_feat, n_dim, K = k, bias = False, cached = True, add_self_loops = False)
		
		# decoder layer
		self.layer1 = nn.Linear(n_dim, n_hid)
		self.layer2 = nn.Linear(n_hid, n_feat,bias = False)

	# def forward(self,data):
	# 	x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
	# 	# encoder
	# 	x = self.gc1(x, edge_index, edge_attr)
	# 	embedding = torch.tanh(x)

	# 	# decoder
	# 	x = self.layer1(embedding)
	# 	x = F.relu(x)
	# 	x = self.layer2(x)
	# 	return x, embedding
	
	# def forward(self,data):
	# 	x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
	# 	# encoder
	# 	x = self.gc1(x, edge_index, edge_attr)
	# 	# row-norm (normalize each-row vector into a sphere)
	# 	embedding = torch.div(x, torch.sqrt(torch.sum(x**2, 1, keepdims=True)))

	# 	# decoder
	# 	x = torch.mm(embedding, embedding.t())
	# 	return x, embedding
	
	def forward(self,data):
		x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
		# encoder
		x = self.gc1(x, edge_index, edge_attr)
		# row-norm
		embedding = torch.div(x, torch.sqrt(torch.sum(x**2, dim=1, keepdims=True)))

		# decoder
		x = F.relu(embedding)
		x = self.layer1(x)
		x = F.relu(x)
		x = self.layer2(x)
		return x, embedding

def partition(algo_type, workload, n_hid, n_dim, k, origin_candidate_length=None, beam_search_width=None):
	torch.manual_seed(seed = 777) # CPU pytorch seed
	random.seed(777)

	if workload.query_num == 0:
		return 0, row.partition(workload)

	# adjacent matrix
	adj = workload.affinity_matrix
	print("number of graph nodes:", adj.shape[0])
	# graph preprocessing
	for i in range(adj.shape[0]):
		adj[i][i] = np.max(adj)
	adj = adj/np.max(adj)

	# initialize node feature as one-hot vector
	features = torch.diag(torch.from_numpy(np.array([1.0 for i in range(adj.shape[1])],dtype=np.float32)))

	# in autoencoder, the output is input
	label_x = copy.deepcopy(features)
	np.set_printoptions(threshold=1e6)

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
	optimizer = optim.Adam(model.parameters(), lr = 1e-5, weight_decay = 5e-4)
	
	min_loss = float('inf')
	# early-stop
	patience = 20
	step = 0
	optimal_model = None

	indices = [i for i in range(features.shape[1])]
	for i in range(1000):
		# early-stop is triggered
		if(step >= patience):
			break

		if len(indices) > 1:
			# randomly split train set and validation set
			train_index = random.sample(indices, max(1, math.ceil((features.shape[1]/4))))
			
			valid_index = []
			for ele in indices:
				if ele not in train_index:
					valid_index.append(ele)
		else: # there is only one node in the graph
			train_index = indices
			valid_index = indices

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
	
	# model = copy.deepcopy(optimal_model)
	model.eval()
	with torch.no_grad():
		output,embedding = model(data)
	embedding = embedding.numpy().astype(np.float64)
	
	# similarity = np.dot(embedding, embedding.T)
	# visualization.heatmap(adj, [str(i) for i in range(adj.shape[0])], [str(i) for i in range(adj.shape[1])], 
	# 					"affinity matrix", "affinity between attributes", "attribute id")
	# visualization.heatmap(similarity, [str(i) for i in range(adj.shape[0])], [str(i) for i in range(adj.shape[1])],
	# 					"similarity matrix", "similarity between learned feature vectors", "learned feature vector id")
	
	if algo_type == "VPGAE-B":
		beam_cost,beam_partitions = beam_search(embedding, workload, origin_candidate_length, beam_search_width)
		
		return beam_cost,beam_partitions
	
	elif algo_type == "VPGAE":
		kmeans_cost,kmeans_partitions = simple_kmeans_search(embedding, workload)

		return kmeans_cost,kmeans_partitions
	else:
		print("no corresponding variant.")