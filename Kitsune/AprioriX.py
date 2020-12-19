import glob
import pandas as pd
import sys
import math
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

    if check_cluster_degeneri(clusters):
        raise Exception("CLUSTER DEGENERE ")

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

        for algorithm in ['KernelKmeans']:
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
            if check_cluster_degeneri(clusters[device.name][algorithm][k.n_clusters]):
                raise Exception("CLUSTER DEGENERE")

    return clusters

os.chdir('./Kitsune')

clusters = load_clusters()

cluster_dataset = []

for dev in Device:
    clustering = clusters[dev.name]['KernelKmeans'][10].values()
    for cluster in clustering:
        cluster_dataset.append(cluster)


t_encoder = TransactionEncoder()
te_dictionary = t_encoder.fit(cluster_dataset).transform(cluster_dataset)
df = pd.DataFrame(te_dictionary, columns = t_encoder.columns_)
print(df)
df_apriori = fpgrowth(df, min_support = 0.05, use_colnames = True)
df_apriori['n_features'] = df_apriori['itemsets'].apply(lambda x: len(x))
df_apriori = df_apriori[df_apriori['n_features'] > 1]
#df_apriori = df_apriori.sort_values('n_features')
#pd.set_option('display.max_rows', df_apriori.shape[0]+1)
print(df_apriori)

solver = AlgorithmX(115)

for index, row in df_apriori.iterrows():
    #print(list(row['itemsets']))
    solver.appendRow(list(row['itemsets']), str(index))

for solution in solver.solve():
    print(solution)
