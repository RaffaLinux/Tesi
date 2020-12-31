import glob
import pandas as pd
import sys
import math
import datetime
import os
import progressbar
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

def generate_graph(graphs_list, device):
    fig = plt.figure()
    ax = plt.gca()
    j = 0
    f1_score_mean=np.zeros(len(graphs_list[device]['Kshape'])+len(graphs_list[device]['KernelKmeans'])+len(graphs_list[device]['Kmeans'])+1)
    f1_score_std=np.zeros(len(graphs_list[device]['Kshape'])+len(graphs_list[device]['KernelKmeans'])+len(graphs_list[device]['Kmeans'])+1)
    for algorithm in ['Kshape','Kmeans','KernelKmeans']:
        for i in graphs_list[dev.name][algorithm]:
            f1_score = np.zeros(10)
            mean_fpr = np.linspace(0,1,100)
            for fold in range(10):
                dataset = pd.read_csv('./SKF/'+dev.name+'/'+algorithm+str(i)+'/SKF'+str(fold)+'.csv')
                dataset = dataset.to_numpy()
                fpr,tpr,thresholds = metrics.roc_curve(dataset[:,1],dataset[:,4])
                indices = np.where(fpr>=0.001)
                index = np.min(indices)
                Soglia = thresholds[index]
                labels = np.zeros(dataset.shape[0])
                for i in range(dataset.shape[0]):
                    if dataset[i,4] <= Soglia:
                        labels[i] = 0
                    else:
                        labels[i] = 1
                f1_score[Fold] = metrics.f1_score(dataset[:,1], labels[:], average = 'macro')
            f1_score_mean = f1_score.mean()
            f1_score_std = f1_score.std()
            f1_mean_score = plt.bar(index, f1_score_mean,.3)
            f1_upper = np.minimum(f1_score_mean + f1_score_std, 1)
            f1_lower = np.maximum(f1_score_mean - f1_score_std, 0)


            
            alg_str_fix = ""
            if algorithm == 'Kshape': alg_str_fix = "K-Shape"
            elif algorithm == 'Kmeans': alg_str_fix = "K-Means"
            elif algorithm == 'KernelKmeans': alg_str_fix = "Kernel K-Means"
            line = plt.plot(mean_fpr, mean_tpr, color = colors[j%len(colors)], label=''+alg_str_fix+' '+str(i),linewidth = 1, linestyle = linestyles[math.floor(j/len(colors))])
            j = j+1


    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.yticks(np.arange(0, 1.05, .1))
    plt.xticks(np.arange(0, 1.1, .1))
    plt.ylabel('F1 Score Mean (Standard Deviation)')
    plt.title('F1 Score Comparison - '+device)
    lgd = plt.figlegend(bbox_to_anchor = (0.50,-.4),ncol= 3, loc = 'lower center', fontsize = 'medium', fancybox = True, frameon = True)
    lgd.get_frame().set_edgecolor('k')

    axins = zoomed_inset_axes(ax,6, loc = 7,bbox_to_anchor = (400,135))
    plt.plot([0, 1], [0, 1], 'k--')


    #plt.tight_layout()
    plt.savefig('./Graphs/ROC/'+device+'.pdf',bbox_extra_artists = [lgd],bbox_inches='tight')




os.chdir('./Kitsune/')
graphs_list = dict()
for dev in Device:
    graphs_list[dev.name] = dict()
    graphs_list[dev.name]['Kshape'] = []
    graphs_list[dev.name]['Kmeans'] = []
    graphs_list[dev.name]['KernelKmeans'] = []

device = Device(int(sys.argv[1])).name

graphs_list[device]['KernelKmeans'] = range(2,19)
generate_graph(graphs_list,device)