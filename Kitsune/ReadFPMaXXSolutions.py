import pickle
import os
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
from tslearn.clustering import TimeSeriesKMeans, KShape,KernelKMeans
from enum import Enum
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth,fpmax


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

        for algorithm in [sys.argv[1]]:
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

    return clusters

algorithm = 'Kshape'

os.chdir('./Kitsune')

clusters = load_clusters()

cluster_dataset = []

for dev in Device:
    clustering = clusters[dev.name][algorithm][10].values()
    for cluster in clustering:
        cluster_dataset.append(cluster)


t_encoder = TransactionEncoder()
te_dictionary = t_encoder.fit(cluster_dataset).transform(cluster_dataset)
df = pd.DataFrame(te_dictionary, columns = t_encoder.columns_)
print(len(df))





os.chdir('./../Kitsune/FPMaXX')
rows = pickle.load(open('rows_'+algorithm,'rb'))
#print(len(rows))
solutions = pickle.load(open('solutions_'+algorithm,'rb'))
print(solutions)