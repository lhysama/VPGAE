import random
import numpy as np

def random_generator(num = 100, q_num_range = [1,20], a_num_range = [1,100], freq_range = [1,10], selectivity_range = [0.0005,0.5],lenth_of_attr_range= [2,50],cardinality_range=[5000,25000]):
	seed = 777
	np.random.seed(seed)
	random.seed(seed)
	dataset = []

	for _ in range(num):
		data = []
		q_num = random.randint(q_num_range[0],q_num_range[1])
		a_num = random.randint(a_num_range[0],a_num_range[1])
		attribute_usage_list = []
		freq_list = []
		scan_key_list = []
		selectivity_list = []

		attribute_list = [i for i in range(1,a_num+1)]
		for t in range(q_num):
			temp = random.randint(1,max(1,int(a_num/q_num)))
			random.shuffle(attribute_list)
			random_attr_select = attribute_list[:temp]
			random_attr_select = sorted(random_attr_select)
			attribute_usage_list.append(random_attr_select)
			rad = random.random()
			if rad < 0.1:
				freq_list.append(random.randint(freq_range[0],freq_range[1]))
			else:
				freq_list.append(random.randint(50,100))
			scan_key_list.append([random_attr_select[random.randint(0,len(random_attr_select)-1)]])
			selectivity_list.append(round(selectivity_range[0] + 0.0005*(random.randint(0,(selectivity_range[1]-selectivity_range[0])/0.0005)),4))
		data.append(q_num)
		data.append(a_num)
		data.append(attribute_usage_list)
		data.append(freq_list)
		data.append(selectivity_list)

		length_of_attributes = []
		for i in range(a_num):
			length_of_attributes.append(random.randint(lenth_of_attr_range[0],lenth_of_attr_range[1]))
		data.append(length_of_attributes)
		data.append(scan_key_list)
		data.append([random.randint(1,a_num_range[1])])
		data.append(random.randint(cardinality_range[0],cardinality_range[1]))

		dataset.append(data)
	
	return dataset


def lineitem(scaleFactor):
	lineitem = [17,16,[[5, 6, 7, 8, 9, 10, 11],[1,6,7,11],[1, 12, 13],[1, 3, 6, 7],[5, 6, 7, 11],[1, 3, 6, 7, 11],[1, 2, 3, 6, 7],[1, 2, 3, 5, 6, 7],[1, 6, 7, 9],[1, 11, 12, 13, 15],[2, 6, 7, 11],[3, 6, 7, 11],[2, 5, 6],[1, 5],[2, 5, 6, 7, 14, 15],[2, 3, 5, 11],[1, 3, 12, 13]],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[9.90E-01,5.35E-01,6.32E-01,1.0,1.98E-02,3.04E-01,1.0,1.0,2.47E-01,5.18E-03,1.25E-02,3.83E-02,1.0,1.0,2.00E-02,1.52E-01,1.0],[4,4,4,4,4,4,4,4,1,1,10,10,10,25,10,44],[[11],[11],[12,13],None,[5,7,11],[11],None,None,[9],[11,12,13,15],[11],[11],None,None,[5,14,15],[11],None],[1,4],scaleFactor * 6000000]
	
	return lineitem

def tpch_workload(scaleFactor):
	# number of queries, number of attrs, referenced attrs, frequency, selectivity, length of attrs, scan key, clustered index, cardinality
	# Coustomer table already
	customer = [8, 8, [[1,7],[1,4],[1,4],[1,4],[1,2,3,4,5,6,8],[1],[1,2],[1,5,6]], [1,1,1,1,1,1,1,1], [2.00E-01,1.0,1.0,1.0,1.0,1.0,1.0,2.55E-01], [4,25,40,4,15,4,10,117],[[7],None,None,None,None,None,None,[5,6]],[1],scaleFactor * 150000]
	# Lineitem table
	lineitem = [17,16,[[5, 6, 7, 8, 9, 10, 11],[1,6,7,11],[1, 12, 13],[1, 3, 6, 7],[5, 6, 7, 11],[1, 3, 6, 7, 11],[1, 2, 3, 6, 7],[1, 2, 3, 5, 6, 7],[1, 6, 7, 9],[1, 11, 12, 13, 15],[2, 6, 7, 11],[3, 6, 7, 11],[2, 5, 6],[1, 5],[2, 5, 6, 7, 14, 15],[2, 3, 5, 11],[1, 3, 12, 13]],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[9.90E-01,5.35E-01,6.32E-01,1.0,1.98E-02,3.04E-01,1.0,1.0,2.47E-01,5.18E-03,1.25E-02,3.83E-02,1.0,1.0,2.00E-02,1.52E-01,1.0],[4,4,4,4,4,4,4,4,1,1,10,10,10,25,10,44],[[11],[11],[12,13],None,[5,7,11],[11],None,None,[9],[11,12,13,15],[11],[11],None,None,[5,14,15],[11],None],[1,4],scaleFactor * 6000000]
	simplify_lineitem = [17,14,[[4, 5, 6, 7, 8, 9, 10],[1,5,6,10],[1, 11, 12],[1, 3, 5, 6],[4, 5, 6, 10],[1, 3, 5, 6, 10],[1, 2, 3, 5, 6],[1, 2, 3, 4, 5, 6],[1, 5, 6, 8],[1, 10, 11, 12, 14],[2, 5, 6, 10],[3, 5, 6, 10],[2, 4, 5],[1, 4],[2, 4, 5, 6, 13, 14],[2, 3, 4, 10],[1, 3, 11, 12]],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[9.90E-01,5.35E-01,6.32E-01,1.0,1.98E-02,3.04E-01,1.0,1.0,2.47E-01,5.18E-03,1.25E-02,3.83E-02,1.0,1.0,2.00E-02,1.52E-01,1.0],[4,4,4,4,4,4,4,1,1,10,10,10,25,10],[[10],[10],[11,12],None,[4,6,10],[10],None,None,[8],[10,11,12,14],[10],[10],None,None,[4,13,14],[10],None],[1,13],scaleFactor * 6000000]
	# Part table already
	part = [8,9,[[1, 3, 5, 6],[1,5],[1,2],[1,5],[1,4,5,6],[1,4,7],[1,4,6,7],[1,2]],[1,1,1,1,1,1,1,1],[3.93E-03,6.68E-03,5.47E-02,1.0,1.49E-01,1.00E-03,2.37E-03, 1.09E-02],[4,55,25,10,25,4,10,4,23],[[5,6],[5],[2],None,[4,5,6],[4,7],[4,6,7],[2]],[1],scaleFactor * 200000]
	# Supplier table already
	supplier = [10,7,[[1,2,3,4,5,6,7],[1,4],[1,4],[1,4],[1,4],[1,4],[1,2,3,5],[1,7],[1,2,3,4],[1,2,4]],[1,1,1,1,1,1,1,1,1,1],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,5.60E-04,1.0,1.0],[4,25,40,4,15,4,101],[None,None,None,None,None,None,None,[7],None,None],[1],scaleFactor * 10000]
	# PartSupp table already
	partsupp = [5,5,[[1,2,4],[1,2,4],[1,2,3,4],[1,2],[1,2,3]],[1,1,1,1,1],[1.0,1.0,1.0,9.99E-01,1.09E-02],[4,4,4,4,199],[None,None,None,[2],[1]],[1,2],scaleFactor * 800000]
	# Orders table already
	orders = [12,9,[[1, 2, 5, 8],[1, 5, 6],[1, 2, 5],[1,2],[1,2,5],[1,5],[1, 2, 5],[1,6],[1, 2, 9],[1, 2, 4, 5],[1,3],[2]],[1,1,1,1,1,1,1,1,1,1,1,1],[4.90E-01,3.75E-02,1.52E-01,1.0,3.04E-01,1.0,3.82E-02,1.0,9.89E-01,5.00E-06,4.87E-01],[4,4,1,4,10,15,15,4,79],[[5],[5],[5],None,[5],None,[5],None,[9],[1],[3],None],[1],scaleFactor * 1500000]
	# Nation table already
	nation = [9,4,[[1,2,3],[1,2,3],[1,2],[1,2,3],[1,2],[1,2],[1,2],[1,2],[1,2]],[1,1,1,1,1,1,1,1,1],[1.0,1.0,0.04,1.0,1.0,1.0,0.04,0.04,0.04],[4,25,4,152],[None,None,[2],None,None,None,[2],[2],[2]],[1],25]
	# Region table already
	region = [3,3,[[1,2],[1,2],[1,2]],[1,1,1],[0.2,0.2,0.2],[4,25,152],[[2],[2],[2]],[1], 5]
	
	return [customer,lineitem,orders,supplier,part,partsupp,nation,region]
	# return [lineitem]

def HAP():
	selectivity = 0.3
	projectivity = 16
	queries_number = 20
	attribute_number = 160
	
	workload = []
	workload.append(queries_number)
	workload.append(attribute_number)

	attribute_list = [i+1 for i in range(attribute_number)]
	attribute_usage_list = []
	scankey_list = []
	# randomly select projectivity attributes
	for i in range(queries_number):
		random.shuffle(attribute_list)
		attribute_usage_list.append(attribute_list[:projectivity])
		scankey_list.append([attribute_list[0]])

	workload.append(attribute_usage_list)
	workload.append([1 for i in range(queries_number)])
	workload.append([selectivity for i in range(queries_number)])
	attr_length_list = [8]
	for i in range(attribute_number-1):
		attr_length_list.append(4)

	workload.append(attr_length_list)
	workload.append(scankey_list)
	workload.append([1])
	workload.append(100000000)

	print(workload)
	return workload
