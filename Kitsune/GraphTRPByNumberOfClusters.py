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
from matplotlib.ticker import MaxNLocator


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

#    print(mean_tpr[mean_fpr[:] == 0.01])

def generate_graph(graphs_list):
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    markerstyles = ['o', 's', 'P']
    for device in Device:
        device = device.name
        fig = plt.figure()
        ax = plt.gca()
        
        tprs = []
        mean_fpr = np.linspace(1e-4,1,10000)
        for fold in range(10):
            dataset = pd.read_csv('./SKF/'+device+'/Base/SKF'+str(fold)+'.csv')
            dataset = dataset.to_numpy()
            fpr,tpr,thresholds= metrics.roc_curve(dataset[:,1],dataset[:,4])
            #plt.plot(fpr,tpr)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.axhline(y=mean_tpr[mean_fpr[:] == 0.01], color='k', linestyle='-', label = 'Time Clusters')

        j = 0
        for algorithm in ['Kshape','Kmeans','KernelKmeans']:
            y_tprs = []
            x_clusters = []

            alg_str_fix = ""
            if algorithm == 'Kshape': alg_str_fix = "K-Shape"
            elif algorithm == 'Kmeans': alg_str_fix = "K-Means"
            elif algorithm == 'KernelKmeans': alg_str_fix = "Kernel K-Means"

            
            for i in graphs_list[device][algorithm]:
                tprs = []
                aucs = np.zeros(10)
                mean_fpr = np.linspace(1e-4,1,10000)
                for fold in range(10):
                    dataset = pd.read_csv('./SKF/'+device+'/'+algorithm+str(i)+'/SKF'+str(fold)+'.csv')
                    dataset = dataset.to_numpy()
                    fpr,tpr,thresholds= metrics.roc_curve(dataset[:,1],dataset[:,4])
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                auc = np.mean(aucs)
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                std_tpr = np.std(tprs, axis=0)
                y_tprs.append(mean_tpr[mean_fpr[:] == 0.01])
                x_clusters.append(i)
            plt.plot(x_clusters,y_tprs,color = colors[j%len(colors)], marker='o', label = alg_str_fix)
            j = j+1


        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylim([0, 1.05])
        plt.yticks(np.arange(0, 1.05, .1))
        #plt.xticks(np.arange(1e-3, 1.1, .1))
        plt.xlabel('Number of Clusters')
        plt.ylabel('True Positive Rate')
        plt.title('Number of Clusters / TPR - '+device.replace('_',' '))
        lgd = plt.figlegend(bbox_to_anchor = (0.50,-0.05),ncol= 5, loc = 'lower center', fontsize = 'x-small', fancybox = True, frameon = True)
        lgd.get_frame().set_edgecolor('k')


        plt.savefig('./Graphs/TPRNclusters/'+device+'.pdf',bbox_extra_artists = [lgd],bbox_inches='tight')




os.chdir('./Kitsune/')
graphs_list = dict()
graphs_list[Device(0).name] = dict()
graphs_list[Device(0).name]['Kshape'] = range(2,14)
graphs_list[Device(0).name]['Kmeans'] = range(2,15)
graphs_list[Device(0).name]['KernelKmeans'] = range(2,15)

graphs_list[Device(1).name] = dict()
graphs_list[Device(1).name]['Kshape'] = range(2,21)
graphs_list[Device(1).name]['Kmeans'] = range(2,21)
graphs_list[Device(1).name]['KernelKmeans'] = range(2,18)

graphs_list[Device(2).name] = dict()
graphs_list[Device(2).name]['Kshape'] = range(2,16)
graphs_list[Device(2).name]['Kmeans'] = range(2,14)
graphs_list[Device(2).name]['KernelKmeans'] = range(2,19)

graphs_list[Device(3).name] = dict()
graphs_list[Device(3).name]['Kshape'] = range(2,15)
graphs_list[Device(3).name]['Kmeans'] = range(2,16)
graphs_list[Device(3).name]['KernelKmeans'] = range(2,22)

graphs_list[Device(4).name] = dict()
graphs_list[Device(4).name]['Kshape'] = range(2,14)
graphs_list[Device(4).name]['Kmeans'] = range(2,15)
graphs_list[Device(4).name]['KernelKmeans'] = range(2,21)

graphs_list[Device(5).name] = dict()
graphs_list[Device(5).name]['Kshape'] = range(2,16)
graphs_list[Device(5).name]['Kmeans'] = range(2,21)
graphs_list[Device(5).name]['KernelKmeans'] = range(2,24)

graphs_list[Device(6).name] = dict()
graphs_list[Device(6).name]['Kshape'] = range(2,16)
graphs_list[Device(6).name]['Kmeans'] = range(2,22)
graphs_list[Device(6).name]['KernelKmeans'] = range(2,31)

graphs_list[Device(7).name] = dict()
graphs_list[Device(7).name]['Kshape'] = range(2,15)
graphs_list[Device(7).name]['Kmeans'] = range(2,14)
graphs_list[Device(7).name]['KernelKmeans'] = range(2,24)

graphs_list[Device(8).name] = dict()
graphs_list[Device(8).name]['Kshape'] = range(2,25)
graphs_list[Device(8).name]['Kmeans'] = range(2,18)
graphs_list[Device(8).name]['KernelKmeans'] = range(2,21)

generate_graph(graphs_list)