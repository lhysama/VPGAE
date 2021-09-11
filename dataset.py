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

def real_system_wide_table():
	lineitem = [10,
				30,
				[[5, 6, 7, 8, 9, 10, 11],[1, 12, 13],[1,6,7,11],[1, 3, 6, 7],[5, 6, 7, 11],[1, 4],[17,20,21,22],[18,19,23,27],[22,25,26,27,28,29,30],[2,14,15,16,24]],
				[5,3,2,1,1,3,1,1,4,5],
				[1,1,1,1,1,1,1,1,1,1],
				[150,500,233,300,4,50,4,100,4,4,500,100,250,4,4,1000,300,25,4,400,4,33,100,55,155,4,4,900,20,4],
				[None,None,None,None,None,None,None,None,None,None],
				[1,4],
				100158]
	
	return lineitem

def tpch_workload(scaleFactor):
	# number of queries, number of attrs, referenced attrs, frequency, selectivity, length of attrs, scan key, clustered index, cardinality
	# Customer table
	customer = [8, 8, [[1,7],[1,4],[1,4],[1,4],[1,2,3,4,5,6,8],[1],[1,2],[1,5,6]], [1,1,1,1,1,1,1,1], [2.00E-01,1.0,1.0,1.0,1.0,1.0,1.0,2.55E-01], [4,25,40,4,15,4,10,117],[[7],None,None,None,None,None,None,[5,6]],[1],scaleFactor * 150000]
	# Lineitem table
	lineitem = [17,16,[[5, 6, 7, 8, 9, 10, 11],[1,6,7,11],[1, 12, 13],[1, 3, 6, 7],[5, 6, 7, 11],[1, 3, 6, 7, 11],[1, 2, 3, 6, 7],[1, 2, 3, 5, 6, 7],[1, 6, 7, 9],[1, 11, 12, 13, 15],[2, 6, 7, 11],[3, 6, 7, 11],[2, 5, 6],[1, 5],[2, 5, 6, 7, 14, 15],[2, 3, 5, 11],[1, 3, 12, 13]],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[9.90E-01,5.35E-01,6.32E-01,1.0,1.98E-02,3.04E-01,1.0,1.0,2.47E-01,5.18E-03,1.25E-02,3.83E-02,1.0,1.0,2.00E-02,1.52E-01,1.0],[4,4,4,4,4,4,4,4,1,1,10,10,10,25,10,44],[[11],[11],[12,13],None,[5,7,11],[11],None,None,[9],[11,12,13,15],[11],[11],None,None,[5,14,15],[11],None],[1,4],scaleFactor * 6000000]
	simplify_lineitem = [17,14,[[4, 5, 6, 7, 8, 9, 10],[1,5,6,10],[1, 11, 12],[1, 3, 5, 6],[4, 5, 6, 10],[1, 3, 5, 6, 10],[1, 2, 3, 5, 6],[1, 2, 3, 4, 5, 6],[1, 5, 6, 8],[1, 10, 11, 12, 14],[2, 5, 6, 10],[3, 5, 6, 10],[2, 4, 5],[1, 4],[2, 4, 5, 6, 13, 14],[2, 3, 4, 10],[1, 3, 11, 12]],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[9.90E-01,5.35E-01,6.32E-01,1.0,1.98E-02,3.04E-01,1.0,1.0,2.47E-01,5.18E-03,1.25E-02,3.83E-02,1.0,1.0,2.00E-02,1.52E-01,1.0],[4,4,4,4,4,4,4,1,1,10,10,10,25,10],[[10],[10],[11,12],None,[4,6,10],[10],None,None,[8],[10,11,12,14],[10],[10],None,None,[4,13,14],[10],None],[1,13],scaleFactor * 6000000]
	# Part table
	part = [8,9,[[1, 3, 5, 6],[1,5],[1,2],[1,5],[1,4,5,6],[1,4,7],[1,4,6,7],[1,2]],[1,1,1,1,1,1,1,1],[3.93E-03,6.68E-03,5.47E-02,1.0,1.49E-01,1.00E-03,2.37E-03, 1.09E-02],[4,55,25,10,25,4,10,4,23],[[5,6],[5],[2],None,[4,5,6],[4,7],[4,6,7],[2]],[1],scaleFactor * 200000]
	# Supplier table
	supplier = [10,7,[[1,2,3,4,5,6,7],[1,4],[1,4],[1,4],[1,4],[1,4],[1,2,3,5],[1,7],[1,2,3,4],[1,2,4]],[1,1,1,1,1,1,1,1,1,1],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,5.60E-04,1.0,1.0],[4,25,40,4,15,4,101],[None,None,None,None,None,None,None,[7],None,None],[1],scaleFactor * 10000]
	# PartSupp table
	partsupp = [5,5,[[1,2,4],[1,2,4],[1,2,3,4],[1,2],[1,2,3]],[1,1,1,1,1],[1.0,1.0,1.0,9.99E-01,1.09E-02],[4,4,4,4,199],[None,None,None,[2],[1]],[1,2],scaleFactor * 800000]
	# Orders table
	orders = [12,9,[[1, 2, 5, 8],[1, 5, 6],[1, 2, 5],[1,2],[1,2,5],[1,5],[1, 2, 5],[1,6],[1, 2, 9],[1, 2, 4, 5],[1,3],[2]],[1,1,1,1,1,1,1,1,1,1,1,1],[4.90E-01,3.75E-02,1.52E-01,1.0,3.04E-01,1.0,3.82E-02,1.0,9.89E-01,5.00E-06,4.87E-01],[4,4,1,4,10,15,15,4,79],[[5],[5],[5],None,[5],None,[5],None,[9],[1],[3],None],[1],scaleFactor * 1500000]
	# Nation table
	nation = [9,4,[[1,2,3],[1,2,3],[1,2],[1,2,3],[1,2],[1,2],[1,2],[1,2],[1,2]],[1,1,1,1,1,1,1,1,1],[1.0,1.0,0.04,1.0,1.0,1.0,0.04,0.04,0.04],[4,25,4,152],[None,None,[2],None,None,None,[2],[2],[2]],[1],25]
	# Region table
	region = [3,3,[[1,2],[1,2],[1,2]],[1,1,1],[0.2,0.2,0.2],[4,25,152],[[2],[2],[2]],[1], 5]
	
	return [customer,lineitem,orders,supplier,part,partsupp,nation,region]
	# return [lineitem]

# TPC-DS benchmark with scaleFactor = 1
def tpcds_workload():
	customer_address = [6, 13, [[1, 9, 11], [1, 9, 10], [1, 10], [1, 9, 11], [1, 7], [1, 12]], [1, 1, 1, 1, 1, 1], [0.25498, 1.0, 1.0, 0.21936, 0.00858, 0.10898], [4, 17, 11, 9, 16, 11, 9, 14, 3, 11, 14, 5, 21], [[9, 11], None, None, [9, 11], [7], [12]], [1], 50000]
	customer_demographics = [7, 9, [[1, 2, 3, 4], [1, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 3, 4], [1], [1, 3, 4]], [1, 1, 1, 1, 1, 1, 1], [0.014285714285714285, 0.08571428571428572, 0.014285714285714285, 0.014285714285714285, 0.08571428571428572, 1.0, 0.05714285714285714], [4, 2, 2, 21, 4, 11, 4, 4, 4], [[2, 3, 4], [3, 4], [2, 3, 4], [2, 3, 4], [3, 4], None, [3, 4]], [1], 1920800]
	date_dim = [23, 28, [[1, 3, 7, 9], [1, 3, 7], [1, 3, 7], [1, 3, 7, 11], [1, 3, 7, 9], [1, 3, 4], [1, 3, 7], [1, 3, 7], [1, 3, 7, 10], [1, 3, 7, 9], [1, 3, 7, 15], [1, 3, 7], [1, 3, 7, 9], [1, 3, 4, 11], [1, 3, 7, 9], [1, 3, 4], [1, 3, 4, 9], [1, 3, 4, 7, 9, 11], [1, 3, 7, 10], [1, 3, 7, 8], [1, 3, 7, 9], [1, 3, 7, 9], [1, 3, 4]], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0.08213664800339499, 0.004996646086873195, 0.004996646086873195, 0.001245739161384824, 0.0004106832400169749, 0.005010335528207094, 0.004996646086873195, 0.004996646086873195, 0.0034497392161425893, 0.0004106832400169749, 0.005010335528207094, 0.004996646086873195, 0.00042437268135087406, 0.004996646086873195, 0.0004106832400169749, 0.004996646086873195, 0.004996646086873195, 0.004996646086873195, 0.0009856397760407399, 0.0021492422894221685, 0.004996646086873195, 0.00042437268135087406, 0.005010335528207094], [4, 17, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 10, 7, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 2], [[9], [7], [7], [7, 11], [7, 9], [4], [7], [7], [7, 10], [7, 9], [7], [7], [7, 9], [4], [7, 9], [4], [4], [4], [7, 10], [7, 8], [7], [7, 9], [4]], [3], 73049]
	warehouse = [2, 14, [[1, 3], [1, 3]], [1, 1], [1.0, 1.0], [4, 17, 18, 4, 11, 9, 16, 11, 9, 18, 3, 11, 14, 5], [None, None], [1], 5]
	ship_mode = [2, 6, [[1, 3], [1, 3]], [1, 1], [1.0, 1.0], [4, 17, 31, 11, 21, 21], [None, None], [1], 20]
	time_dim = [1, 10, [[1, 3, 4, 5]], [1], [0.020833333333333332], [4, 17, 4, 4, 4, 4, 3, 21, 21, 21], [[4, 5]], [3], 86400]
	reason = [1, 3, [[1, 3]], [1], [0.0], [4, 17, 101], [[3]], [1], 35]
	income_band = [1, 3, [[1, 2, 3]], [1], [0.2], [4, 4, 4], [[2, 3]], [1], 20]
	item = [13, 22, [[1, 8, 9, 14, 15], [1, 2], [1, 8, 9, 14, 15, 21], [1, 9, 11, 13, 22], [1, 2], [1, 2], [1, 12, 13, 21], [1, 8, 9, 21], [1, 9, 11, 13, 14, 15], [1, 8, 9, 21], [1, 9, 11, 13, 21], [1, 9, 11, 13, 22], [1, 9, 11, 13]], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0.0006111111111111111, 1.0, 0.016277777777777776, 1.0, 1.0, 1.0, 0.017833333333333333, 0.017833333333333333, 0.05188888888888889, 0.00938888888888889, 0.05188888888888889, 1.0, 0.09211111111111112], [4, 17, 4, 4, 102, 6, 6, 4, 51, 4, 51, 4, 51, 4, 51, 21, 21, 21, 11, 11, 4, 51], [[14, 15], None, [21], None, None, None, [21], [21], [9, 11, 13], [21], [9, 11, 13], None, [11, 13]], [1], 18000]
	store = [13, 29, [[1], [1, 26], [1, 25], [1, 24], [1, 2, 6, 28], [1], [1], [1], [1, 2], [1, 24], [1, 7, 23], [1, 6, 18], [1, 6]], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0.4166666666666667, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9166666666666666, 1.0, 0.08333333333333333], [4, 17, 4, 4, 4, 5, 4, 4, 21, 13, 4, 8, 56, 15, 4, 8, 4, 8, 3, 7, 16, 11, 7, 18, 3, 11, 14, 5, 5], [None, None, [25], [24], [28], None, None, None, None, [24], [7], None, [6]], [1], 12]
	call_center = [2, 31, [[1, 2, 7, 12], [1, 7]], [1, 1], [0.5, 1.0], [4, 17, 4, 4, 4, 4, 12, 6, 4, 4, 21, 13, 4, 51, 70, 13, 4, 4, 4, 51, 11, 11, 16, 11, 7, 18, 3, 11, 14, 5, 5], [None, None], [1], 6]
	customer = [7, 18, [[1, 5], [1, 5], [1, 8, 9, 10, 11], [1, 8, 9, 10, 11], [1, 9, 10], [2, 3, 4, 5, 9, 10], [1, 3, 4, 5]], [1, 1, 1, 1, 1, 1, 1], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [4, 17, 4, 4, 4, 4, 4, 11, 21, 31, 2, 4, 4, 4, 9, 56, 51, 4], [None, None, None, None, None, None, None], [1], 100000]
	web_site = [1, 26, [[1, 5]], [1], [1.0], [4, 17, 4, 4, 7, 4, 4, 8, 13, 4, 33, 69, 13, 4, 51, 11, 10, 16, 11, 7, 18, 3, 11, 14, 5, 4], [None], [1], 30]
	store_returns = [2, 20, [[5], [3, 9, 10, 11]], [1, 1], [1.0, 0.4166684057124175], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6], [None, None], [10], 287514]
	household_demographics = [7, 5, [[1, 4], [1, 3, 4, 5], [1, 3, 4, 5], [1, 4, 5], [1, 2], [1, 3], [1, 4]], [1, 1, 1, 1, 1, 1, 1], [0.2, 0.1111111111111111, 0.1111111111111111, 0.85, 1.0, 0.16666666666666666, 0.1], [4, 4, 16, 4, 4], [[4], [3, 4, 5], [3, 4, 5], [4, 5], None, [3], [4]], [1], 7200]
	web_page = [0, 14, [], [], [], [4, 17, 4, 4, 4, 4, 2, 4, 19, 51, 4, 4, 4, 4], [], [1], 60]
	promotion = [2, 19, [[1, 10, 15], [1, 10, 15]], [1, 1], [0.9966666666666667, 0.9966666666666667], [4, 17, 4, 4, 4, 5, 4, 51, 2, 2, 2, 2, 2, 2, 2, 2, 41, 16, 2], [[10, 15], [10, 15]], [1], 300]
	catalog_page = [0, 9, [], [], [], [4, 17, 4, 4, 11, 4, 4, 75, 8], [], [1], 11718]
	inventory = [1, 4, [[1, 2, 4]], [1], [1.0], [4, 4, 4, 4], [None], [3], 11745000]
	catalog_returns = [1, 27, [[1, 8, 12, 27]], [1], [1.0], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6], [None], [17], 144067]
	web_returns = [0, 24, [], [], [], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6], [], [14], 71763]
	web_sales = [1, 34, [[1, 3, 14, 15, 16]], [1], [1.0], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 7, 6, 3, 6, 6, 6, 7, 7, 6], [None], [18], 719384]
	catalog_sales = [3, 34, [[1, 4, 22], [1, 5, 16, 17, 19, 21, 22, 28], [1, 3, 12, 14, 15]], [1, 1, 1], [1.0, 1.0, 1.0], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 7, 6, 3, 6, 6, 6, 7, 7, 6], [None, None, None], [18], 1441548]
	store_sales = [19, 23, [[1, 3, 14], [1, 3, 5, 9, 11, 13, 14, 20], [1, 5, 6, 7, 8, 11, 14, 16, 17, 23], [1, 3, 4, 8, 16], [1, 3, 5, 8, 11, 13, 14, 20], [1, 4, 6, 8, 10], [1, 3, 16], [1, 8, 14], [1, 5, 7, 8, 11, 14, 23], [1, 3, 16], [1, 3, 8, 14], [1, 3, 16], [1, 3, 8, 14], [1, 3, 8, 11, 14], [1, 4, 6, 8, 10], [1, 4, 6, 7, 8, 10, 20, 23], [1, 3, 8, 14], [3, 4, 10, 11, 14], [2, 6, 8]], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1.0, 1.0, 0.030935243806077203, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1502379527316307, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24997569785349555, 1.0], [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 3, 6, 6, 7, 6, 3, 6, 6, 6], [None, None, [14, 23], None, None, None, None, None, [14, 23], None, None, None, None, None, None, None, None, None, None], [10], 2880404]
	
	return [customer_address,customer_demographics,date_dim,warehouse,ship_mode,time_dim,reason,income_band,
			item,store,call_center,customer,web_site,store_returns,household_demographics,web_page,promotion,
			catalog_page,inventory,catalog_returns,web_returns,web_sales,catalog_sales,store_sales]


def HAP(queries_number_list):
	dataset = []
	for queries_number in queries_number_list:
		selectivity = 0.3
		projectivity = 16
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
		dataset.append(workload)

	return dataset

def dynamic_workloads(group_num=3, queries_number=15):
	seed = 777
	np.random.seed(seed)
	random.seed(seed)
	
	dynamic_workloads_group = []

	selectivity_range = [0, 1.0]
	projectivity = 3
	attribute_number = 100

	for g in range(group_num):
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
		workload.append([random.randint(1,10) for i in range(queries_number)])
		workload.append([min(selectivity_range)+(max(selectivity_range)-min(selectivity_range))*np.random.uniform() for i in range(queries_number)])
		attr_length_list = [8]
		for i in range(attribute_number-1):
			attr_length_list.append(4)

		workload.append(attr_length_list)
		workload.append(scankey_list)
		workload.append([1])
		workload.append(100000)
		dynamic_workloads_group.append(workload)
	
	# Combine multiple dynamic workloads into one static workload
	static_workload=[]
	static_workload.append(queries_number*group_num)
	static_workload.append(attribute_number)
	static_workload.append([])
	static_workload.append([])
	static_workload.append([])
	static_workload.append([])
	static_workload.append([])
	static_workload[5] = dynamic_workloads_group[0][5]
	for workload in dynamic_workloads_group:
		static_workload[2] += workload[2]
		static_workload[3] += workload[3]
		static_workload[4] += workload[4]
		static_workload[6] += workload[6]
	static_workload.append(dynamic_workloads_group[0][7])
	static_workload.append(dynamic_workloads_group[0][8])

	return dynamic_workloads_group, static_workload