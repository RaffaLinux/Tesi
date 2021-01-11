import glob
import pandas as pd
import sys
import math
import datetime
import os
import progressbar
import seaborn as sns
import numpy as np
import tensorflow as tf
import time
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from enum import Enum
import pickle
from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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

def compute_accuracy(graphs_list, device):
    df = pd.DataFrame(columns= ['Algorithm','Device','F1 Mean', 'F1 Std. Dev.'])

    for algorithm in ['Kshape','Kmeans','KernelKmeans']:

        for i in graphs_list[device][algorithm]:
            f1_score = np.zeros(10)
            for fold in range(10):
                dataset = pd.read_csv('./SKF/'+device+'/'+algorithm+str(i)+'/SKF'+str(fold)+'.csv')
                dataset = dataset.to_numpy()
                fpr,tpr,thresholds = metrics.roc_curve(dataset[:,1],dataset[:,4])
                indices = np.where(fpr>=0.001)
                index = np.min(indices)
                soglia = thresholds[index]
                labels = np.zeros(dataset.shape[0])
                for j in range(dataset.shape[0]):
                    if dataset[j,4] <= soglia:
                        labels[j] = 0
                    else:
                        labels[j] = 1
                f1_score[fold]=metrics.f1_score(dataset[:,1],labels[:],average='macro')

            f1_mean = f1_score.mean()
            f1_std = f1_score.std()
            alg_str_fix = ""
            if algorithm == 'Kshape': alg_str_fix = "K-Shape "
            elif algorithm == 'Kmeans': alg_str_fix = "K-Means "
            elif algorithm == 'KernelKmeans': alg_str_fix = "Kernel K-Means "            
            df.loc[len(df)] = [alg_str_fix+str(i), device.replace('_',' ').title(), f1_mean,f1_std]

    f1_score = np.zeros(10)
    for fold in range(10):
        dataset = pd.read_csv('./SKF/'+device+'/Base/SKF'+str(fold)+'.csv')
        dataset = dataset.to_numpy()
        fpr,tpr,thresholds = metrics.roc_curve(dataset[:,1],dataset[:,4])
        indices = np.where(fpr>=0.001)
        index = np.min(indices)
        soglia = thresholds[index]
        labels = np.zeros(dataset.shape[0])
        for j in range(dataset.shape[0]):
            if dataset[j,4] <= soglia:
                labels[j] = 0
            else:
                labels[j] = 1
        f1_score[fold]=metrics.f1_score(dataset[:,1],labels[:],average='macro')

    f1_mean = f1_score.mean()
    f1_std = f1_score.std()         
    df.loc[len(df)] = ['Time Clusters', device.replace('_',' ').title(), f1_mean,f1_std]
    generate_graph(df,device)


def generate_graph(df,device):
    sns.set_style("ticks")
    g = sns.catplot(data=df, kind="bar", x="Device",ci = 'F1 Std. Dev.', y="F1 Mean", hue="Algorithm",palette="tab10", alpha=1, height=5, aspect = 4/3)
    #g.despine(left=True)
    g.set_axis_labels("", "F1 Score")
    g.legend.set_title("")
    g._legend.remove()

    ax = g.facet_axis(0,0)
    for index, p in enumerate(ax.patches):
        algorithms = ["K-Shape", "K-Means","Kernel K-Means", "Time Clusters"]
        df_temp = df.loc[df["Algorithm"].str.contains(algorithms[index % 4])]
        yerr = df_temp.iloc[0]["F1 Std. Dev."]
        ax.errorbar(p.get_x() +0.1, p.get_height(), yerr, ecolor = 'k', linewidth = .5)

        ax.text(p.get_x() + 0.05, 
                p.get_height() * 1.02, 
                '{0:.2f}'.format(p.get_height()), 
                color='black', rotation='horizontal', size= "medium")
    plt.legend(loc='lower center', ncol= 4, bbox_to_anchor = (.5,-.3), fancybox = True,edgecolor = "k")
    #plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
    g.savefig('./Graphs/F1Scores/'+device+'.pdf')




os.chdir('./Kitsune/')
graphs_list = dict()
for dev in Device:
    graphs_list[dev.name] = dict()
    graphs_list[dev.name]['Kshape'] = []
    graphs_list[dev.name]['Kmeans'] = []
    graphs_list[dev.name]['KernelKmeans'] = []


for index, best_clusters in enumerate([[12,11,14],[5,17,17], [9,5,11],[10,7,11],[13,7,15],[14,13,6],[16,12,30],[11,8,20],[10,18,14]]):
    device = Device(index).name
    graphs_list[device]['Kmeans'] = [best_clusters[0]]
    graphs_list[device]['Kshape'] = [best_clusters[1]]
    graphs_list[device]['KernelKmeans'] = [best_clusters[2]]
    compute_accuracy(graphs_list,device)

