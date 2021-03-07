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

def compute_accuracy(graphs_list):

    df = pd.DataFrame(columns= ['Approach','Device','F1 Mean', 'F1 Std. Dev.'])

    for approach, paths in graphs_list.items():
        len_dataset = 0
        f1_score = []
        if approach == "Centralized - Time Clusters":
            for fold in range(10):
                dataset = pd.read_csv('./SKF/Centralized/Base/SKF'+str(fold)+'.csv')
                dataset = dataset.to_numpy()
                fpr,tpr,thresholds = metrics.roc_curve(dataset[:,1],dataset[:,4])
                indices = np.where(fpr>=0.001)
                index = np.min(indices)
                soglia = thresholds[index]
                labels = np.zeros(dataset.shape[0])
                len_dataset = len_dataset + len(dataset[:,1])
                for j in range(dataset.shape[0]):
                    if dataset[j,4] <= soglia:
                        labels[j] = 0
                    else:
                        labels[j] = 1
                f1_score.append(metrics.f1_score(dataset[:,1],labels[:],average='macro'))

            f1_mean = np.mean(f1_score)
            f1_std = np.std(f1_score)
            df.loc[len(df)] = [approach, "All Devices",f1_mean, f1_std]
        else:
            for device in Device:
                device = device.name
                for fold in range(10):
                    if approach == "Distributed - Time Clusters":
                        dataset = pd.read_csv('./SKF/'+device+'/Base/SKF'+str(fold)+'.csv')
                    else:
                        if device in paths[0]:
                            dataset = pd.read_csv('./SKF/'+paths[0]+'/SKF'+str(fold)+'.csv')
                        elif device in paths[1]:
                            dataset = pd.read_csv('./SKF/'+paths[1]+'/SKF'+str(fold)+'.csv')
                        else:   
                            dataset = pd.read_csv('./SKF/'+device+'/Base/SKF'+str(fold)+'.csv') 
                    dataset = dataset.to_numpy()
                    fpr,tpr,thresholds = metrics.roc_curve(dataset[:,1],dataset[:,4])
                    indices = np.where(fpr>=0.001)
                    index = np.min(indices)
                    soglia = thresholds[index]
                    labels = np.zeros(dataset.shape[0])
                    len_dataset = len_dataset + len(dataset[:,1])
                    for j in range(dataset.shape[0]):
                        if dataset[j,4] <= soglia:
                            labels[j] = 0
                        else:
                            labels[j] = 1
                    f1_score.append(metrics.f1_score(dataset[:,1],labels[:],average='macro'))

   
            df.loc[len(df)] = [approach, "All Devices",np.mean(f1_score), np.std(f1_score)]



    generate_graph(df)

def generate_graph(df):
    sns.set_style("ticks")
    g = sns.catplot(data=df, kind="bar", x="Device", y="F1 Mean",ci = "F1 Std. Dev.", hue="Approach",palette="tab10", alpha=1, height=5, aspect = 4/3)
    #g.despine(left=True)
   #g.despine(left=True)
    g.set_axis_labels("", "F1 Score")
    g.legend.set_title("")
    g._legend.remove()

    ax = g.facet_axis(0,0)
    for index, p in enumerate(ax.patches):
        algorithms = ["Hybrid - Good", "Hybrid - Bad","Distributed - Time Clusters", "Centralized - Time Clusters"]
        df_temp = df.loc[df["Approach"].str.contains(algorithms[index % 4])]
        yerr = df_temp.iloc[0]["F1 Std. Dev."]
        ax.errorbar(p.get_x() +0.1, p.get_height(), yerr, ecolor = 'k', linewidth = .5)

        ax.text(p.get_x() + 0.04, 
                p.get_height() * 1.02, 
                '{0:.2f}'.format(p.get_height()), 
                color='black', rotation='horizontal', size= "large")
    plt.ylim((0,1))
    plt.legend(loc='lower center', ncol= 2, bbox_to_anchor = (.5,-.2), fancybox = True,edgecolor = "k", fontsize = 'medium')
    #plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
    g.savefig('./Graphs/F1Scores/HybridAllDevices.pdf')




os.chdir('./Kitsune/')
graphs_list = dict()


graphs_list["Hybrid - Good"] = ["Hybrid/Ecobee_ThermostatSimpleHome_XCS7_1003_WHT_Security_Camera", "Hybrid/Ennio_DoorbellProvision_PT_737E_Security_CameraProvision_PT_838_Security_CameraSimpleHome_XCS7_1002_WHT_Security_Camera"]
graphs_list["Hybrid - Bad"] = ["Hybrid/Provision_PT_737E_Security_CameraSimpleHome_XCS7_1003_WHT_Security_Camera", "Hybrid/Danmini_DoorbellEcobee_ThermostatPhilips_B120N10_Baby_MonitorSamsung_SNH_1011_N_Webcam"]
graphs_list["Distributed - Time Clusters"] = []
graphs_list["Centralized - Time Clusters"] = []
compute_accuracy(graphs_list)