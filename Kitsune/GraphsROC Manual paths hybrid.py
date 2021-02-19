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

def generate_graph(graphs_list, graph_name):
    fig = plt.figure()
    ax = plt.gca()
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    #plt.rc('axes', prop_cycle=(cycler('color', colors)*cycler('linestyle,[-,--,') ))
    
    ax.set_xscale("symlog",linthreshx=0.001)
    x_coinflip = np.linspace(1e-4,1,1000)
    y_coinflip = np.linspace(1e-4,1,1000)

    plt.plot([1e-2, 1e-2], [0, 10], 'k:', alpha = .3)

    plt.plot(x_coinflip,y_coinflip, 'k--', label= "Chance")


    j = 0
    np.set_printoptions(threshold=sys.maxsize)

 #GOOD CLUSTERS   
    tprs = []
    aucs = []
    mean_fpr = np.linspace(1e-4,1,10000)
    for dev in Device:
        for fold in range(10):

            if dev.value in [1,8]:
                dataset = pd.read_csv('./SKF/Hybrid/Ecobee_ThermostatSimpleHome_XCS7_1003_WHT_Security_Camera/SKF'+str(fold)+'.csv')
            if dev.value in [2,4,5,7]:
                dataset = pd.read_csv('./SKF/Hybrid/Ennio_DoorbellProvision_PT_737E_Security_CameraProvision_PT_838_Security_CameraSimpleHome_XCS7_1002_WHT_Security_Camera/SKF'+str(fold)+'.csv')
            else:
                dataset = pd.read_csv('./SKF/'+dev.name+'/Base/SKF'+str(fold)+'.csv')

            dataset = dataset.to_numpy()
            dataset = dataset[(dataset[:,2] == dev.value)]
            fpr,tpr,thresholds= metrics.roc_curve(dataset[:,1],dataset[:,4])
            aucs.append(metrics.roc_auc_score(dataset[:,1],dataset[:,4], max_fpr=0.01))
            #plt.plot(fpr,tpr)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
    auc = np.mean(aucs)
    print(auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    print(mean_tpr[mean_fpr[:] == 0.01])


    line = plt.plot(mean_fpr, mean_tpr, color = colors[j%len(colors)], label= "Hybrid good clusters",linewidth = 1, linestyle = linestyles[math.floor(j/len(colors))])
    j = j+1


#BAD CLUSTERS
    tprs = []
    aucs = []
    mean_fpr = np.linspace(1e-4,1,10000)
    for dev in Device:
        for fold in range(10):

            if dev.value in [4,8]:
                dataset = pd.read_csv('./SKF/Hybrid/Provision_PT_737E_Security_CameraSimpleHome_XCS7_1003_WHT_Security_Camera/SKF'+str(fold)+'.csv')
            if dev.value in [0,1,3,6]:
                dataset = pd.read_csv('./SKF/Hybrid/Danmini_DoorbellEcobee_ThermostatPhilips_B120N10_Baby_MonitorSamsung_SNH_1011_N_Webcam/SKF'+str(fold)+'.csv')
            else:
                dataset = pd.read_csv('./SKF/'+dev.name+'/Base/SKF'+str(fold)+'.csv')

            dataset = dataset.to_numpy()
            dataset = dataset[(dataset[:,2] == dev.value)]
            fpr,tpr,thresholds= metrics.roc_curve(dataset[:,1],dataset[:,4])
            aucs.append(metrics.roc_auc_score(dataset[:,1],dataset[:,4], max_fpr=0.01))

            #plt.plot(fpr,tpr)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
    auc = np.mean(aucs)
    print(auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    print(mean_tpr[mean_fpr[:] == 0.01])


    line = plt.plot(mean_fpr, mean_tpr, color = colors[j%len(colors)], label= "Hybrid bad clusters",linewidth = 1, linestyle = linestyles[math.floor(j/len(colors))])
    j = j+1

#DISTRIBUTED CLUSTERS
    tprs = []
    aucs = []
    mean_fpr = np.linspace(1e-4,1,10000)
    for dev in Device:
        for fold in range(10):

            dataset = pd.read_csv('./SKF/'+dev.name+'/Base/SKF'+str(fold)+'.csv')

            dataset = dataset.to_numpy()
            dataset = dataset[(dataset[:,2] == dev.value)]
            fpr,tpr,thresholds= metrics.roc_curve(dataset[:,1],dataset[:,4])
            aucs.append(metrics.roc_auc_score(dataset[:,1],dataset[:,4], max_fpr=0.01))

            #plt.plot(fpr,tpr)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
    auc = np.mean(aucs)
    print(auc)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    print(mean_tpr[mean_fpr[:] == 0.01])


    line = plt.plot(mean_fpr, mean_tpr, color = colors[j%len(colors)], label= "Distributed Time Clusters",linewidth = 1, linestyle = linestyles[math.floor(j/len(colors))])
    j = j+1

#CENTRALIZED
    tprs = []
    mean_fpr = np.linspace(1e-4,1,10000)
    for fold in range(10):
        dataset = pd.read_csv('./SKF/Centralized/Base/SKF'+str(fold)+'.csv')
        dataset = dataset.to_numpy()
        fpr,tpr,thresholds= metrics.roc_curve(dataset[:,1],dataset[:,4])
        #plt.plot(fpr,tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)



    line = plt.plot(mean_fpr, mean_tpr, color = colors[j%len(colors)], label= "Centralized Time Clusters",linewidth = 1, linestyle = linestyles[math.floor(j/len(colors))])
    j = j+1


    plt.xlim([1e-4, 1.05])
    plt.ylim([0, 1.05])
    plt.yticks(np.arange(0, 1.05, .1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve '+graph_name)
    lgd = plt.figlegend(bbox_to_anchor = (0.50,-.20),ncol= 2, loc = 'lower center', fontsize = 'x-small', fancybox = True, frameon = True)
    lgd.get_frame().set_edgecolor('k')


    plt.savefig('./Graphs/ROC/'+graph_name+'.pdf',bbox_extra_artists = [lgd],bbox_inches='tight')




os.chdir('./Kitsune/')
graphs_list = dict()

# graphs_list["Hybrid - Cluster 0"] = "Hybrid/Ecobee_ThermostatSimpleHome_XCS7_1003_WHT_Security_Camera"
# graphs_list["Centralized - K-Means - Best 10 mean "] = "Centralized/Kmeans/best_10_mean"
# graphs_list["Distributed - SimpleHome 1003 - K-Shape 18"] = "SimpleHome_XCS7_1003_WHT_Security_Camera/Kshape18"
# graphs_list["Distributed - Ecobee Thermostat - K-Means 5"] = "Ecobee_Thermostat/Kmeans5"
# generate_graph(graphs_list,"Hybrid Clustering - Cluster 0")

# graphs_list = dict()

# graphs_list["Hybrid - Cluster 1"] = "Hybrid/Ennio_DoorbellProvision_PT_737E_Security_CameraProvision_PT_838_Security_CameraSimpleHome_XCS7_1002_WHT_Security_Camera"
# graphs_list["Centralized - K-Means - Best 10 mean "] = "Centralized/Kmeans/best_10_mean"
# graphs_list["Distributed - Provision 737E - Kernel K-Means 15"] = "Provision_PT_737E_Security_Camera/KernelKmeans15"
# graphs_list["Distributed - Provision 838 - K-Shape 13"] = "Provision_PT_838_Security_Camera/Kshape13"
# graphs_list["Distributed - SimpleHome 1002 - K-Means 11"] = "SimpleHome_XCS7_1002_WHT_Security_Camera/Kmeans11"
# generate_graph(graphs_list,"Hybrid Clustering - Cluster 1")

graphs_list = dict()

graphs_list["Hybrid - Cluster 0"] = "Hybrid/Ecobee_ThermostatSimpleHome_XCS7_1003_WHT_Security_Camera"
graphs_list["Hybrid - Cluster 1"] = "Hybrid/Ennio_DoorbellProvision_PT_737E_Security_CameraProvision_PT_838_Security_CameraSimpleHome_XCS7_1002_WHT_Security_Camera"
graphs_list["Centralized - K-Means - Best 10 mean "] = "Centralized/Kmeans/best_10_mean"
generate_graph(graphs_list,"Hybrid Clustering - Means")
