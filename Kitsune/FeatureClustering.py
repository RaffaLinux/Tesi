import glob
import pandas as pd
import sys
import math
import os
import datetime
import progressbar
import numpy as np
import time
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from sklearn.preprocessing import MinMaxScaler
from tslearn.preprocessing import TimeSeriesScalerMinMax
from matplotlib import pyplot as plt
from matplotlib.pyplot import legend
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


def Clustering(dataset, algorithm = 'Kmeans', device = "Ennio_Doorbell"):
   results = pd.DataFrame(columns= dataset.columns[:115])


   dataset = pd.concat([dataset], ignore_index=True)

#~400 entries limite superiore per KernelKmeans
   if algorithm == 'KernelKmeans':
      subset = dataset.head(n = 400)
   else:
      subset = dataset.head(n = 2048)

   # subset = subset.transpose()
   subset = subset.to_numpy().astype('float32')
   

   if not os.path.isdir('./Clustering/'+device):
      os.makedirs('./Clustering/'+device)


   
   benchmark = open('./Clustering/'+device+'/benchmark-'+algorithm+'.txt', 'a+')
   benchmark.write('\nBenchmark ' + datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
   benchmark.close()


   for i in [5,10,15,20]:
      wall_time = time.time()
      process_time = time.process_time()

      clustering = subset[:, :116]
      clustering_features = clustering[:,:115]
      scaler_clustering = MinMaxScaler(feature_range = (0,1))
      clustering_features = scaler_clustering.fit_transform(clustering_features)
      clustering_features = np.transpose(clustering_features)
      # print(clustering_features)

      # clustering = subset[:116, :]
      # clustering_features = clustering[:115,:]
      # scaler_clustering = TimeSeriesScalerMinMax(value_range = (0,1))
      # clustering_features = scaler_clustering.fit_transform(clustering_features)

      if algorithm == 'Kmeans':
         k = TimeSeriesKMeans(n_clusters=i, metric="softdtw", max_iter=3, verbose=1, n_jobs = 12, max_iter_barycenter = 3)
         #k = TimeSeriesKMeans.from_json('./Clustering/'+device+'/'+algorithm+str(i)+'.json')
      elif algorithm == 'Kshape':
         k = KShape(n_clusters=i,max_iter= 3, verbose = 1)
         #k = KShape.from_json('./Clustering/'+device+'/'+algorithm+str(i)+'.json')
      elif algorithm == 'KernelKmeans':
         k = KernelKMeans(n_clusters = i, max_iter = 3, n_jobs = 12, verbose = 1)
         #k = KernelKMeans.from_json('./Clustering/'+device+'/'+algorithm+str(i)+'.json')
      else:
         return

      y_pred = k.fit_predict(clustering_features)
      k.to_json('./Clustering/'+device+'/'+algorithm+str(i)+'.json')
      #y_pred = k.predict(clustering_features)
      print("A loop has been completed")

      save_image(device, algorithm, i, k, y_pred, clustering_features)

      wall_clustering_time = time.time() - wall_time
      process_clustering_time = time.process_time() - process_time
      benchmark = open('./Clustering/'+device+'/benchmark-'+algorithm+'.txt', 'a+')
      benchmark.write('\n'+str(i)+ ' clusters')
      benchmark.write('\n     Wall time: ' + str(wall_clustering_time))
      benchmark.write('\n     Process time: ' + str (process_clustering_time))
      benchmark.close()

   
   results.loc[len(results)] = k.labels_.tolist()
   print(results)
   results.to_csv('./Clustering/'+device+'/'+algorithm+'.csv', index = False, sep = ',')
   
   return results


def save_image(device, algorithm, i, k, y_pred, clustering_features):
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
   plt.savefig('./Clustering/'+device+'/'+algorithm+str(i)+'.svg', bbox_extra_artists = extra_artists, bbox_inches = 'tight')#, dpi = 300)



#Lettura del dataset dai csv da far diventare una funzione
os.chdir('./Kitsune')
dataset = dict()
dataset_path = "./Dataset/"
progress = 0
csv_paths = glob.glob(dataset_path+'**/*.csv', recursive = True)
if len(sys.argv) > 1:
   csv_paths = [i for i in csv_paths if Device(int(sys.argv[1])).name in i]
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
   progress += 1
   bar.update(progress)
bar.finish()



#Conversione dataset Ennio da Dataframe a numpy, creazione dataset benigno, maligno e misto

device = Device(int(sys.argv[1])).name
device_dataset = dataset[device]
device_benign = device_dataset['benign_traffic']
dataset = device_benign

dataset = pd.concat([dataset], ignore_index=True)
print(dataset)


for algorithm in ['Kshape']:
   Clustering(dataset = dataset,algorithm = algorithm, device = device)
