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
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from enum import Enum
import json

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

CLUSTERS_MAX = 40

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
   return dataset

def save_image(device, algorithm, i, k, y_pred, clustering_features):
   
#~400 entries limite superiore per KernelKmeans   
   redline = None
   plt.figure(figsize= (15,15))
   sz = clustering_features.shape[1]
   extra_artists = []

   for yi in range(i):
      plt.subplot(i, 1, 1 + yi,)
      for xx in clustering_features[y_pred == yi]:
         plt.plot(xx.ravel(), "k-", alpha=.2)
      if algorithm == 'Kmeans':
         redline = plt.plot(k.cluster_centers_[yi].ravel(), "r-", linewidth = 0.65)
      elif algorithm == 'Kshape':
         scaler = MinMaxScaler(feature_range = (0,1))
         centroids = scaler.fit_transform(k.cluster_centers_[yi])
         #centroids = k.cluster_centers_[yi]
         redline = plt.plot(centroids.ravel(), "r-", linewidth = 0.65)
      plt.xlim(0, sz)
      plt.ylim(0, 1)
      extra_artists.append(plt.text(1.01, 0.50,'Cluster %d' % (yi),transform=plt.gca().transAxes, fontsize = 'large'))

   if algorithm == 'Kmeans':
      stl = plt.suptitle('K-means', fontsize = 'xx-large')
   elif algorithm == 'Kshape':
      stl = plt.suptitle('K-Shape', fontsize = 'xx-large')
   else:
         stl = plt.suptitle('Kernel K-means', fontsize = 'xx-large')

   if algorithm != 'KernelKmeans':
      if algorithm == 'Kmeans':
         lgd = plt.figlegend((redline), ["Cluster centroid"],bbox_to_anchor = (0.5,-0.04), loc = 'lower center', fontsize = 'x-large', fancybox = True, frameon = True)
      elif algorithm == 'Kshape':
         lgd = plt.figlegend((redline), ["Cluster centroid (scaled)"],bbox_to_anchor = (0.5,-0.04), loc = 'lower center', fontsize = 'x-large', fancybox = True, frameon = True)
      lgd.get_frame().set_edgecolor('k')
      extra_artists.append(lgd)

   extra_artists.append(stl)
   plt.tight_layout()
   plt.savefig('./Clustering/'+device+'/Reclustering/'+algorithm+'/'+str(i)+'.svg', bbox_extra_artists = extra_artists, bbox_inches = 'tight')#, dpi = 300)


def labels_to_clusters(labels):
   clusters = dict()
   for i in range(len(labels)):
      if labels[i] not in clusters.keys():
         clusters[labels[i]] = [i]
      else:
         clusters[labels[i]].append(i)
   return clusters

def progressive_clustering(device, algorithm, dataset, initial_clusters):

      k = Clustering(dataset, i, algorithm, device)

      clusters_temp = labels_to_clusters(k.labels_)
      print(clusters_temp)
      if not check_cluster_degeneri(clusters_temp):
         new_clusters = clusters_temp
         k.to_json('./Clustering/'+device+'/Reclustering/'+algorithm+'/'+str(i)+'.json')
      else:
         print("\nALGORITHM: "+algorithm+" CLUSTERS:" + str(initial_clusters)+" "+device+" HA CLUSTER DEGENERI. RICALCOLARE")


def Clustering(dataset, n_clusters, algorithm = 'Kmeans', device = "Ennio_Doorbell"):

   dataset = pd.concat([dataset], ignore_index=True)

#~400 entries limite superiore per KernelKmeans
   if algorithm == 'KernelKmeans':
      subset = dataset.head(n = 400)
   else:
      subset = dataset.head(n = 2048)

   subset = subset.to_numpy().astype('float32')  

   if not os.path.isdir('./Clustering/'+device+'/Reclustering/'+algorithm):
      os.makedirs('./Clustering/'+device+'/Reclustering/'+algorithm)

   clustering = subset[:, :116]
   clustering_features = clustering[:,:115]
   scaler_clustering = MinMaxScaler(feature_range = (0,1))
   clustering_features = scaler_clustering.fit_transform(clustering_features)
   clustering_features = np.transpose(clustering_features)


   if algorithm == 'Kmeans':
      k = TimeSeriesKMeans(n_clusters=n_clusters, metric="softdtw", max_iter=3, verbose=1, n_jobs = 12, max_iter_barycenter = 3)
   elif algorithm == 'Kshape':
      k = KShape(n_clusters=n_clusters,max_iter= 3, verbose = 1)
   elif algorithm == 'KernelKmeans':
      k = KernelKMeans(n_clusters = n_clusters, max_iter = 3, n_jobs = 12, verbose = 1)
   else:
      return

   y_pred = k.fit_predict(clustering_features)
   save_image(device, algorithm, n_clusters, k, y_pred, clustering_features)

   print("A loop has been completed")
   
   return k


def check_cluster_degeneri(clusters):
   d_clusters = 0
   for i in clusters.copy().keys():
      if len(clusters[i]) == 1:
         d_clusters += 1
   return d_clusters > 0



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


def load_features_clusters(device):
   clusters_path = './Clustering/'
   clusters_path = clusters_path + device + '/'
   df_kshape = pd.read_csv(clusters_path+'Kshape.csv', delimiter = ',')
   df_kernel = pd.read_csv(clusters_path+'KernelKmeans.csv', delimiter = ',')
   df_kmeans = pd.read_csv(clusters_path+'Kmeans.csv', delimiter = ',')
   return [df_kmeans, df_kshape, df_kernel]


#Conversione dataset Ennio da Dataframe a numpy, creazione dataset benigno, maligno e misto
device = Device(0)
dataset = dict()

if len(sys.argv) > 1:
   device = Device(int(sys.argv[1]))
   dataset = load_dataset(dataset, device)
else:
   print(sys.argv)
   sys.exit('Add a device as argument (0...9) and algorithm to launch this script')

device = device.name
device_dataset = dataset[device]
device_benign = device_dataset['benign_traffic']
dataset = device_benign

dataset = pd.concat([dataset], ignore_index=True)

features_clusters = dict()
features_clusters['Kshape'] = {}
features_clusters['KernelKmeans'] = {}
features_clusters['Kmeans'] = {}
dataframes = load_features_clusters(device)


dataframe_to_feature_clusters(dataframes[0], features_clusters['Kmeans'], device, 'Kmeans')
dataframe_to_feature_clusters(dataframes[1], features_clusters['Kshape'], device, 'Kshape')
dataframe_to_feature_clusters(dataframes[2], features_clusters['KernelKmeans'],device, 'KernelKmeans')

n_clusters_lower = int(sys.argv[3])
n_clusters_upper = int(sys.argv[4])
algorithm = sys.argv[2]

for i in range(n_clusters_lower,n_clusters_upper):
   if check_cluster_degeneri(features_clusters[algorithm][20]) == False and i == 20:
      print('Skipping '+algorithm+' '+str(i)+': clusters already generated')
      continue
   if check_cluster_degeneri(features_clusters[algorithm][15]) == False and i == 15:
      print('Skipping '+algorithm+' '+str(i)+': clusters already generated')
      continue
   if check_cluster_degeneri(features_clusters[algorithm][10]) == False and i == 10:
      print('Skipping '+algorithm+' '+str(i)+': clusters already generated')
      continue
   if check_cluster_degeneri(features_clusters[algorithm][5]) == False and i == 5:
      print('Skipping '+algorithm+' '+str(i)+': clusters already generated')
      continue
   
   progressive_clustering(device,algorithm,dataset,i)
   




