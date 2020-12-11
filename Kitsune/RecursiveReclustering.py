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
from tslearn.clustering import TimeSeriesKMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from enum import Enum

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


#Lettura del dataset dai csv
def load_dataset(dataset, dev: Device = None):
   os.chdir('./Kitsune')
   dataset = dict()
   dataset_path = "./Dataset/"
   progress = 0
   csv_paths = glob.glob(dataset_path+'**/*.csv', recursive = True)
   if dev is not None:
      csv_paths = [i for i in csv_paths if dev.name in i]
   print("Loading dataset")
   bar = progressbar.ProgressBar(maxval=len(csv_paths), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
   bar.start()
   for csv_path in csv_paths:
   #print(csv_path)
      attack = ''
      device = csv_path.split('\\')[1]
      if device not in dataset:
         dataset[device] = {}
      if( len(csv_path.split('\\')) == 4):
         attack = csv_path.split('\\')[2]+'_'+csv_path.split('\\')[3]
      else:
         attack = csv_path.split('\\')[2]
      attack = attack.replace(".csv","")
      #print(attack)
      dataset[device][attack] = pd.read_csv(csv_path, delimiter = ',')
      dataset[device][attack]['Attack'],dataset[device][attack]['Device'] = [Attack[attack].value,Device[device].value]
      progress += 1
      bar.update(progress)
   bar.finish()

def reclustering(clusters, device, algorithm):
   n_dClusters = 0 #Numero di clusters degeneri

   for i in clusters.copy().keys():
      if len(clusters[i]) <= 1: #Ovviamente è == 1 in pratica, ma why not
         n_dClusters += 1

   if n_dClusters == 0: return
   

def dataframe_to_feature_clusters(dataframe, subset, device, algorithm):
   subset[5] = dict()
   subset[10] = dict()
   subset[15] = dict()
   subset[20] = dict()
   
   df_list = dataframe.values.tolist()
   for i in range(len(df_list[0])):
      if df_list[0][i] not in subset[5].keys():
         subset[5][df_list[0][i]] = [i]
      else:
         subset[5][df_list[0][i]].append(i)

   for i in range(len(df_list[1])):
      if df_list[1][i] not in subset[10].keys():
         subset[10][df_list[1][i]] = [i]
      else:
         subset[10][df_list[1][i]].append(i)

   for i in range(len(df_list[2])):
      if df_list[2][i] not in subset[15].keys():
         subset[15][df_list[2][i]] = [i]
      else:
         subset[15][df_list[2][i]].append(i)
   
   for i in range(len(df_list[3])):
      if df_list[3][i] not in subset[20].keys():
         subset[20][df_list[3][i]] = [i]
      else:
         subset[20][df_list[3][i]].append(i)

   reclustering(subset[5], device, algorithm)
   reclustering(subset[10], device, algorithm)
   reclustering(subset[15], device, algorithm)
   reclustering(subset[20], device, algorithm)


def load_features_clusters(features_clusters, device: Device):
   clusters_path = './Clustering/'
   clusters_path = clusters_path + device.name + '/'
   df_kshape = pd.read_csv(clusters_path+'Kshape.csv', delimiter = ',')
   df_kernel = pd.read_csv(clusters_path+'KernelKmeans.csv', delimiter = ',')
   df_kmeans = pd.read_csv(clusters_path+'Kmeans.csv', delimiter = ',')
   features_clusters['Kshape'] = {}
   features_clusters['KernelKmeans'] = {}
   features_clusters['Kmeans'] = {}
   dataframe_to_feature_clusters(df_kshape, features_clusters['Kshape'], device, 'Kshape')
   dataframe_to_feature_clusters(df_kernel, features_clusters['KernelKmeans'],device, 'KernelKmeans')
   dataframe_to_feature_clusters(df_kmeans, features_clusters['Kmeans'], device, 'Kmeans')





#Conversione dataset Ennio da Dataframe a numpy, creazione dataset benigno, maligno e misto
device = Device(0)
device_dataset = dict()
if len(sys.argv) > 2:
   device = Device(int(sys.argv[1]))
   load_dataset(device_dataset, device)
else:
   print(sys.argv)
   sys.exit('Add a device as argument (0...9) and algorithm to launch this script')

features_clusters = dict()
load_features_clusters(features_clusters, device)

device_benign = device_dataset['benign_traffic']
device_benign = device_benign.sample(frac = 1) #ELIMINARE IN FASE FINALE
device_benign = pd.concat([device_benign], ignore_index=True)
print(device_benign)
device_benign = device_benign.to_numpy()
device_benign = device_benign.astype('float32')


device_malign = pd.concat([value for key, value in device_dataset.items() if key not in ('benign_traffic')], ignore_index=True)
device_malign = device_malign.sample(frac = 0.25) #ELIMINARE IN FASE FINALE
device_malign = device_malign.to_numpy().astype('float32')
device_benign = np.concatenate([device_benign], axis = 0)
device_all = np.concatenate([device_benign,device_malign], axis = 0)

