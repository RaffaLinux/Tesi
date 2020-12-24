import glob
import pandas as pd
import sys
import math
import pickle
import datetime
import os
import progressbar
import numpy as np
import time
from pandas import DataFrame
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, KShape

from enum import Enum
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth,fpmax
from algorithm_x import AlgorithmX


class Device(Enum):
    Danmini_Doorbell = 0
    Ecobee_Thermostat = 1
    Ennio_Doorbell = 2
    Philips_B120N10_Baby_Monitor = 3
    Provision_PT_737E_Security_Camera = 4
    Provision_PT_838_Security_Camera = 5
    Samsung_SNH_1011_N_Webcam = 6
    SimpleHome_XCS7_1002_WHT_Security_Camera = 7
    SimpleHome_XCS7_1003_WHT_Security_Camera = 8

class Attack(Enum):
    benign_traffic = 0
    gafgyt_combo = 1
    gafgyt_junk = 2
    gafgyt_scan = 3
    gafgyt_tcp = 4
    gafgyt_udp = 5
    mirai_ack = 6
    mirai_scan = 7
    mirai_syn = 8
    mirai_udp = 9
    mirai_udpplain = 10

def labels_to_clusters(labels):
    clusters = dict()

    for i in range(len(labels)):

        if labels[i] not in clusters.keys():
            clusters[labels[i]] = [i]
        else:
            clusters[labels[i]].append(i)

#    if check_cluster_degeneri(clusters):
#       raise Exception("CLUSTER DEGENERE ")

    return clusters


def check_cluster_degeneri(clusters):
    d_clusters = 0

    for i in clusters.copy().keys():
        if len(clusters[i]) == 1:
            d_clusters += 1

    return d_clusters > 0

def load_clusters():
    clusters = dict()
    clusters_path = './FinalClustering/'

    for device in Device:
        clusters[device.name] = dict()

        for algorithm in ['Kmeans']:
            clusters[device.name][algorithm] = dict()
            json_paths_1 = glob.glob(clusters_path+device.name+'/'+algorithm+'/*.json')

        for json_path in json_paths_1:
            if algorithm == 'Kshape':
                k = KShape.from_json(json_path)
            elif algorithm == 'KernelKmeans':
                k = KernelKMeans.from_json(json_path)
            elif algorithm == 'Kmeans':
                k = TimeSeriesKMeans.from_json(json_path)
            
            clusters[device.name][algorithm][k.n_clusters] = dict()
            clusters[device.name][algorithm][k.n_clusters] = labels_to_clusters(k.labels_)
            # if check_cluster_degeneri(clusters[device.name][algorithm][k.n_clusters]):
            #     raise Exception("CLUSTER DEGENERE")

    return clusters

def invert_dict(d): 
    inverse = dict() 
    for key in d: 
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse: 
                # If not create a new list
                inverse[item] = [key] 
            else: 
                inverse[item].append(key) 
    return inverse

os.chdir('./Kitsune')

clusters = load_clusters()

cluster_dataset = []

for dev in Device:
    clustering = clusters[dev.name][sys.argv[1]][10].values()
    for cluster in clustering:
        cluster_dataset.append(cluster)


t_encoder = TransactionEncoder()
te_dictionary = t_encoder.fit(cluster_dataset).transform(cluster_dataset)
df = pd.DataFrame(te_dictionary, columns = t_encoder.columns_)
print(df)

rows_List= []
features = set(range(0,115))
for i in [9,8,7,6,5,4,3,2,1]:
    if len(features) == 0:
        break
    min_support = i/df.shape[0] - 0.00001
    df_apriori = fpmax(df, min_support = min_support, use_colnames = True)

    df_apriori['n_features'] = df_apriori['itemsets'].apply(lambda x: len(x))
    df_apriori = df_apriori[df_apriori['n_features'] > 3] #Kshape >3, Kmeans >3
    print(df_apriori)
    delete_set = set()
    #Create X dict
    for index,row in df_apriori.iterrows():
        row_set = set(row['itemsets'])
        intersection = row_set.intersection(features)
        if len(intersection) > 0:
            rows_List.append((list(row_set), row['support']))
    delete_set = delete_set.union(row_set)
    print(len(rows_List))
    #print(rows_List)
    features = features.difference(delete_set)
    
#print(rows_List)

print('\n\n')



solver = AlgorithmX(115)

for i,row in enumerate(rows_List):
    #print(list(row['itemsets']))
    solver.appendRow(row[0], str(i))

biggest_clusters_weighted = ()
biggest_clusters_mean = ()
best_support_weighted = ()
best_support_mean = ()
best_10_mean = ()
best_10_weighted = ()

possible_solutions = solver.solve()
for solution in possible_solutions:
    if(len(solution) < 200):
        support_score_mean = 0
        support_score_weighted = 0
        for elem in solution:
            support_score_mean += rows_List[int(elem)][1]
            support_score_weighted += (rows_List[int(elem)][1]*len(rows_List[int(elem)][0]))
        support_score_mean /= len(solution)
        support_score_weighted /= 115

        if biggest_clusters_mean == ():
            biggest_clusters_mean = (solution,support_score_mean)
            biggest_clusters_weighted = (solution,support_score_weighted)
            best_support_mean = (solution,support_score_mean)
            best_support_weighted = (solution,support_score_weighted)

        

        else:
            if (support_score_mean > best_support_mean[1]) or (support_score_mean == best_support_mean[1] and len(solution) < len(best_support_mean[0])):
                best_support_mean = (solution, support_score_mean)
                print("New Best support mean. Len = "+str(len(solution))+" Support score = "+str(support_score_mean))
            
            if (support_score_weighted > best_support_weighted[1]) or (support_score_weighted == best_support_weighted[1] and len(solution) < len(best_support_weighted[0])):
                best_support_weighted = (solution, support_score_weighted)
                print("New Best support weighted. Len = "+str(len(solution))+" Support score = "+str(support_score_weighted))


            if (len(solution) < len(biggest_clusters_mean[0])) or (len(solution) == len(biggest_clusters_mean[0]) and support_score_mean > biggest_clusters_mean[1]):
                biggest_clusters_mean = (solution, support_score_mean)
                print("New biggest clusters mean. Len = "+str(len(solution))+" Support score = "+str(support_score_mean))


            if (len(solution) < len(biggest_clusters_weighted[0])) or (len(solution) == len(biggest_clusters_weighted[0]) and support_score_weighted > biggest_clusters_weighted[1]):
                biggest_clusters_weighted = (solution, support_score_weighted)
                print("New biggest clusters weighted. Len = "+str(len(solution))+" Support score = "+str(support_score_weighted))

            if (best_10_mean == () and len(solution) == 10):
                best_10_mean = (solution, support_score_mean)
                print("New best 10 mean Len. = "+str(len(solution))+" Support score = "+str(support_score_mean))
            elif (len(solution) == 10 and support_score_mean > best_10_mean[1]):
                best_10_mean = (solution, support_score_mean)
                print("New best 10 mean Len. = "+str(len(solution))+" Support score = "+str(support_score_mean))

            if (best_10_weighted == () and len(solution) == 10):
                best_10_weighted= (solution, support_score_weighted)
                print("New best 10 weighted. Len = "+str(len(solution))+" Support score = "+str(support_score_weighted))
            elif (len(solution) == 10 and support_score_weighted > best_10_weighted[1]):
                best_10_weighted= (solution, support_score_weighted)
                print("New best 10 weighted. Len = "+str(len(solution))+" Support score = "+str(support_score_weighted))

            

        #print((solution,support_level))


solutions = [biggest_clusters_weighted,biggest_clusters_mean,best_support_weighted,best_support_mean, best_10_mean, best_10_weighted]
print(solutions)
filename = 'solutions_'+sys.argv[1]
outfile = open(filename,'wb')
pickle.dump(solutions,outfile)
outfile.close()

filename1 = 'rows_'+sys.argv[1]
outfile1 = open(filename1,'wb')
pickle.dump(rows_List,outfile1)
outfile1.close()


# Y = {}
# #Create Y dict
# for index, row in df_apriori.iterrows():
#     Y[index] = list(row['itemsets'])

# solution = []

# solve(X,Y,solution)
# print(solution)
# print(X)
# print(Y)