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

def compute_accuracy():


    for algorithm in ['KernelKmeans','Kmeans','Kshape']:
        df = pd.DataFrame(columns= ['Method','Attack','Accuracy Mean', 'Accuracy Std. Dev.'])

        score_means = []
        score_devs = []
        for method in ['best_10_mean','best_10_weighted','best_support_mean','best_support_weighted','biggest_clusters_mean','biggest_clusters_weighted']:
            for atk in Attack:
                acc_score = np.zeros(10)
                for fold in range(10):
                    dataset = pd.read_csv('./SKF/Centralized/'+algorithm+'/'+method+'/SKF'+str(fold)+'.csv')
                    dataset = dataset.to_numpy()
                    fpr,tpr,thresholds = metrics.roc_curve(dataset[:,1],dataset[:,4])
                    indices = np.where(fpr>=0.1)
                    index = np.min(indices)
                    soglia = thresholds[index]
                    dataset = dataset[(dataset[:,3] == atk.value)]
                    labels = np.zeros(dataset.shape[0])
                    for j in range(dataset.shape[0]):
                        if dataset[j,4] <= soglia:
                            labels[j] = 0
                        else:
                            labels[j] = 1
                    labels = np.concatenate([labels, [0,1]])
                    if(atk.value == 0):
                        acc_score[fold] = (np.unique(labels,return_counts= True)[1][0] - 1 )/len(dataset[:,1])
                    else:
                        acc_score[fold] = (np.unique(labels,return_counts= True)[1][1] - 1 )/len(dataset[:,1])
                acc_mean = acc_score.mean()
                acc_std = acc_score.std()
                meth_str_fix = method.replace("_", " ").capitalize()
                df.loc[len(df)] = [meth_str_fix, atk.name.replace('_',' ').capitalize(), acc_mean,acc_std]

        for atk in Attack:
            acc_score = np.zeros(10)
            for fold in range(10):
                dataset = pd.read_csv('./SKF/Centralized/Base/SKF'+str(fold)+'.csv')
                dataset = dataset.to_numpy()
                fpr,tpr,thresholds = metrics.roc_curve(dataset[:,1],dataset[:,4])
                indices = np.where(fpr>=0.1)
                index = np.min(indices)
                soglia = thresholds[index]
                dataset=dataset[(dataset[:,3] == atk.value)]
                labels = np.zeros(dataset.shape[0])
                for j in range(dataset.shape[0]):
                    if dataset[j,4] <= soglia:
                        labels[j] = 0
                    else:
                        labels[j] = 1
                labels = np.concatenate([labels, [0,1]])
                if(atk.value == 0):
                    acc_score[fold] = (np.unique(labels,return_counts= True)[1][0] - 1)/len(dataset[:,1])
                else:
                    acc_score[fold] = (np.unique(labels,return_counts= True)[1][1] - 1)/len(dataset[:,1])
            acc_mean = acc_score.mean()
            acc_std = acc_score.std()            
            df.loc[len(df)] = ['Time Clusters', atk.name.replace('_',' ').capitalize(), acc_mean,acc_std]

        generate_graph(df, algorithm)

def generate_graph(df, algorithm):
    sns.set_style("ticks")
    g = sns.catplot(data=df, kind="bar", x="Attack", y="Accuracy Mean", hue="Method",palette="tab10", alpha=1, height=5, aspect = 16/9)
    #g.despine(left=True)
    g.set_axis_labels("", "Accuracy")
    g.legend.set_title("")
    g.set_xticklabels(rotation=30)
    g._legend.remove()

    ax = g.facet_axis(0,0)
    for index, p in enumerate(ax.patches):
        methods = ['best_10_mean','best_10_weighted','best_support_mean','best_support_weighted','biggest_clusters_mean','biggest_clusters_weighted']
        df_temp = df.loc[df["Method"].str.contains(methods[index % 6].replace("_", " ").capitalize())]

        ax.text(p.get_x()-0.03,p.get_height() * 1.02, '{0:.2f}'.format(p.get_height()), color='black',rotation = 45, size= "x-small")
    plt.legend(loc='lower center', ncol= 4, bbox_to_anchor = (.5,-.3), fancybox = True,edgecolor = "k", fontsize = 'large')
    #plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
    g.savefig('./Graphs/Accuracy/Centralized/'+algorithm+'.pdf')




os.chdir('./Kitsune/')
graphs_list = dict()

compute_accuracy()