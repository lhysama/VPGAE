import matplotlib.pyplot as plt
import random

from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn import metrics
from sklearn.decomposition import PCA
from pyvis.network import Network


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def PCAVisualization(embedding,partition):
	# PCA the embedding vector to 2 dim
	pca=PCA(n_components=2,whiten=True)
	node_embeddings_2d=pca.fit_transform(embedding)
	
	attr_color = dict()
	color_list = []

	# generate different color
	for i in range(len(partition)):
		color = randomcolor()
		# if color already exists in color_list, change it to another one
		while color in color_list:
			color = randomcolor()
		color_list.append(color)	

	for i,fragment in enumerate(partition):
		for attr in fragment:
			attr_color[attr-1] = color_list[i]

	attr_color = dict(sorted(attr_color.items(),key = lambda x:x[0]))
	nodes_color = list(attr_color.values())

	plt.figure()
	plt.scatter(node_embeddings_2d[:,0], node_embeddings_2d[:,1],c=nodes_color,cmap="jet")
	
	plt.show()

def nxVisualization(partition,label_adj):
	attr_color = dict()
	color_list = []

	# generate different color
	for i in range(len(partition)):
		color = randomcolor()
		# if color already exists in color_list, change it to another one
		while color in color_list:
			color = randomcolor()
		color_list.append(color)

	for i,fragment in enumerate(partition):
		for attr in fragment:
			attr_color[attr-1] = color_list[i]
			
	attr_color = dict(sorted(attr_color.items(),key = lambda x:x[0]))
	nodes_color = list(attr_color.values())

	net = Network(height='100%', width='100%', bgcolor='#222222',font_color='white')

	net.add_nodes([i for i in range(label_adj.shape[0])],color = nodes_color)

	for i in range(label_adj.shape[0]):
		for j in range(label_adj.shape[1]):
			if label_adj[i][j] > 0 and i != j:
				net.add_edge(i,j)
	
	net.show("example.html")