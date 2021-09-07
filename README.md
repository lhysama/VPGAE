# Learning Vertical Partitions with Graph AutoEncoder

## Introduction

This is a PyTorch implementation of our proposed vertical partitioning algorithm: **VPGAE** and **VPGAE-B**, as described in our paper:

**VPGAE: Learning Vertical Partitions with Graph AutoEncoder** (The paper link will be displayed after publication)

## Requirements

- PyTorch (>=1.4.0)
- PyTorch Geometric
- numpy
- matplotlib
- sklearn
- more_itertools
- scipy
- tqdm
- pyvis

## Run basic experiments

We have implemented all the experiments mentioned in our paper in the *vertical_partition.py* file, so we can directly run this file using Python:

```python
python vertical_partition.py
```

By default, this program will run TPC-H benchmark experiments. You can open the corresponding code blocks in the file to perform other basic experiments.

## **Supplementary experiments (new)**

### 1.  Experiments results on Real Databases:

**(1) Experimental Settings**: 

DBMS: PostgreSQL 11.2

Dataset: Table W with 30 attributes and 100158 lines

Workload: 26 queries generated from template “select ai,...,am from W”

Methods: We use all baseline methods as well as our VPGAE and VPGAE-B to partition W vertically, execute the workload on the partitioned sub-tables, and record the actual execution time of the workload as the real cost. In order to eliminate interference factors such as cache, all experiments are repeated five times after the cache is hot, and the average execution time is taken as the final experimental result.

**(2) Experimental Results** (note: NAVATHE spent more than 1 hour on generating partitioning scheme, we gave up the evaluation of this method):

| Method    | Estimated cost | Workload run time (real cost) |
| --------- | :------------: | :---------------------------: |
| VPGAE-B   |     53804      |          1698.483 ms          |
| HILLCLIMB |     53804      |          1767.468 ms          |
| VPGAE     |     53844      |          1752.324 ms          |
| HYRISE    |     53844      |          1876.993 ms          |
| COLUMN    |     54491      |          2432.890 ms          |
| O2P       |     55946      |          2489.374 ms          |
| ROW       |     331812     |          6233.532 ms          |
| NAVATHE   |     **-**      |             **-**             |

**(3) Performance Analysis**: 

The results show that the real cost trends are consistent with the estimated cost trends, i.e., methods with larger estimated cost are tend to have larger run time. And although VPGAE-B has the same estimated cost with HILLCLIMB, the real cost of VPGAE-B is smaller than HILLCLIMB, and the same phenomenon occurs between VPGAE and HYRISE. We analyzed the partitioning scheme of different methods and found that different schemes might have the same estimated cost. But VPGAE(-B) utilizes the valuable information from the affinity graph, leading to better partitioning scheme with less run time cost. This result indicates that our cost model is reasonable and our partitioning method is effective in reality. 

### 2.  Experiments results on Dynamic Workload:

**(1) Experimental Settings**: 

We first generate 150 queries on table W, and divided them into 15 workloads evenly. Then we take the 15 workloads as streaming input into the partitioning algorithms and evaluate the estimated cost. We also treat all 150 queries as a static workload.

**(2) Experimental results**:

| Method  | Estimated cost on dynamic workload | Estimated cost on static workload |
| ------- | ---------------------------------- | --------------------------------- |
| VPGAE-B | 38330                              | 49480                             |
| VPGAE   | 38890                              | 49640                             |

**(3) Performance Analysis:** 

The results show that both VPGAE and VPGAE-B fit dynamic workload well. Compared with static workload, VPGAE and VPGAE-B on dynamic workloads improves query performance by 27.64% (not the 27.31% mentioned in the feedback) and 29.09%, respectively. This verified that our proposed solution can adapt to workload changes well.

## 3.  Experiments results on TPC-DS benchmark:

**(1) Experimental Settings:**

We generate 1 GB TPC-DS data in **PostgreSQL 11.2** and select 26 template to generate queries. For each template, we only generate one query , and we treat those 26 queries as a workload.

**(2)  Experimental results:**

The experiments results of estimated cost of baselines and our approaches on TPC-DS benchmark are shown in Table 1-4:

Table 1:

| Method    | customer_address | customer_demographics | date_dim | warehouse | ship_mode |
| --------- | ---------------- | --------------------- | -------- | --------- | --------- |
| VPGAE-B   | 193              | 10792                 | 1007     | 2         | 2         |
| HILLCLIMB | 193              | 10792                 | 1007     | 2         | 2         |
| VPGAE     | 193              | 10836                 | 1007     | 2         | 2         |
| HYRISE    | 193              | 11080                 | 1116     | 2         | 2         |
| COLUMN    | 196              | 12208                 | 1300     | 4         | 4         |
| O2P       | 222              | 11920                 | 1334     | 2         | 2         |
| ROW       | 1092             | 19698                 | 5014     | 2         | 2         |
| NAVATHE   | 222              | 11920                 | 1194     | 2         | 2         |

Table 2:

| Method    | income_band | item | call_center | customer | web_site | household_demographics |
| --------- | ----------- | ---- | ----------- | -------- | -------- | ---------------------- |
| VPGAE-B   | 1           | 743  | 2           | 944      | 1        | 35                     |
| HILLCLIMB | 1           | 743  | 2           | 944      | 1        | 35                     |
| VPGAE     | 1           | 750  | 2           | 944      | 1        | 35                     |
| HYRISE    | 1           | 743  | 3           | 944      | 1        | 44                     |
| COLUMN    | 3           | 759  | 6           | 1028     | 2        | 44                     |
| O2P       | 2           | 764  | 3           | 984      | 1        | 41                     |
| ROW       | 1           | 2899 | 2           | 4137     | 1        | 49                     |
| NAVATHE   | 1           | 755  | 2           | 984      | 1        | 41                     |

Table 3:

| Method    | web_page | promotion | catalog_page | inventory | catalog_returns | web_returns |
| --------- | -------- | --------- | ------------ | --------- | --------------- | ----------- |
| VPGAE-B   | 0        | 2         | 0            | 4588      | 78              | 0           |
| HILLCLIMB | 0        | 2         | 0            | 4588      | 78              | 0           |
| VPGAE     | 0        | 2         | 0            | 4588      | 78              | 0           |
| HYRISE    | 0        | 2         | 0            | 4588      | 78              | 0           |
| COLUMN    | 0        | 6         | 0            | 6882      | 123             | 0           |
| O2P       | 0        | 4         | 0            | 4588      | 93              | 0           |
| ROW       | 0        | 4         | 0            | 5735      | 458             | 0           |
| NAVATHE   | 0        | 4         | 0            | 4588      | 93              | 0           |

Table 4 (NAVATHE needs more than 3 hours to generate partitioning scheme on web_sales and catalog_sales tables):

| Method    | web_sales | catalog_sales | store_sales | time_dim | reason | store_returns | store |
| --------- | --------- | ------------- | ----------- | -------- | ------ | ------------- | ----- |
| VPGAE-B   | 422       | 3486          | 52287       | 43       | 1      | 198           | 13    |
| HILLCLIMB | 422       | 3486          | 52287       | 43       | 1      | 198           | 13    |
| VPGAE     | 422       | 3557          | 54956       | 43       | 1      | 198           | 13    |
| HYRISE    | 422       | 3486          | 55525       | 43       | 1      | 198           | 22    |
| COLUMN    | 705       | 4687          | 56940       | 68       | 2      | 285           | 26    |
| O2P       | 705       | 4192          | 57075       | 68       | 1      | 226           | 14    |
| ROW       | 2986      | 17949         | 153672      | 226      | 1      | 1432          | 13    |
| NAVATHE   | -         | -             | 55805       | 51       | 1      | 198           | 13    |

The unnecessary data read and normalized reconstruction joins of baselines and our approaches are shown in Table 5:

Table 5:

|                                 | VPGAE-B | HILLCLIMB | VPGAE  | HYRISE | COLUMN | O2P  | ROW    | NAVATHE |
| ------------------------------- | ------- | --------- | ------ | ------ | ------ | ---- | ------ | ------- |
| Unnecessary data read           | 11.48%  | 11.48%    | 9.49%  |        | 0%     |      | 77.47% |         |
| Normalized reconstruction joins | 45.12%  | 45.12%    | 58.36% |        | 100%   |      | 0%     |         |

**(3) Performance Analysis:**

The results show that both VPGAE and VPGAE-B still maintain good performance when we extended the analysis to TPC-DS.

## Implemented methods

- VPGAE: Our proposed vertical partitioning method using graph model.
- VPGAE-B: A refined version of VPGAE.  We apply a beam search algorithm on the basis of VPGAE to get better solutions.
- HILLCLIMB: A bottom-up greedy search algorithm. It starts from COLUMN layout and in each iteration, it merges two partitions that provide the highest improvement.
- OPTIMAL: An exhaustive algorithm that checks all possible partitioning schemes.
- COLUMN: A special layout that treats one attribute as one partition.
- ROW: A special layout that treats all attributes as one partition. 

Note that, we did not implement NAVATHE, HYRISE and O2P in our project, because these three algorithms have been implemented in [Vertical partitioning algorithms used in physical design of databases](https://github.com/palatinuse/database-vertical-partitioning) and we used the partition results generated by their project for comparison.
