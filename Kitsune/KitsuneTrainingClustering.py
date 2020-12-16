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
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, KShape
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

def StampaValori(device,algorithm, n_clusters,iteration):
   dataset=pd.DataFrame({'Indice': test_index,'Maligno': test_labels[:,0], 'Dispositivo': test_labels[:,2], 'TipologiaAttacco': test_labels[:,1],'RMSE:': RMSE})

   if not os.path.isdir('./Testing/'+device+'/'+algorithm+str(n_clusters)):
      os.makedirs('./Testing/'+device+'/'+algorithm+str(n_clusters))

   dataset.to_csv('./Testing/'+device+'/'+algorithm+str(n_clusters)+'/TSS'+str(iteration+1)+'.csv',index=False, sep=',')

def StampaValoriKFold(device,algorithm,n_clusters,tss_iteration,skf_iteration):
   dataset=pd.DataFrame({'Maligno': test_labels_skf[:,0],'Dispositivo': test_labels_skf[:,2], 'TipologiaAttacco': test_labels_skf[:,1],'RMSE:': RMSE})

   if not os.path.isdir('./Testing/'+device+'/'+algorithm+str(n_clusters)):
      os.makedirs('./Testing/'+device+'/'+algorithm+str(n_clusters))

   dataset.to_csv('./Testing/'+device+'/'+algorithm+str(n_clusters)+'/TSS'+str(tss_iteration+1)+'Fold'+str(skf_iteration)+'.csv',index=False, sep=',')




#Lettura del dataset dai csv
def load_dataset(dev: Device = None):
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
      dataset[device][attack]['Malign'],dataset[device][attack]['Attack'],dataset[device][attack]['Device'] = [0 if Attack[attack].value == 0 else 1,Attack[attack].value,Device[device].value]
      progress += 1
      bar.update(progress)
   bar.finish()
   return dataset

def load_clusters():
   clusters = dict()
   clusters_path = './FinalClustering/'
   for device in Device:
      clusters[device.name] = dict()
      for algorithm in ['Kshape','KernelKmeans', 'Kmeans']:
         clusters[device.name][algorithm] = dict()
         json_paths_1 = glob.glob(clusters_path+device.name+'/Reclustering/'+algorithm+'/*.json')
         
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

def labels_to_clusters(labels):
   clusters = dict()
   for i in range(len(labels)):
      if labels[i] not in clusters.keys():
         clusters[labels[i]] = [i]
      else:
         clusters[labels[i]].append(i)
   return clusters



#Conversione dataset Ennio da Dataframe a numpy, creazione dataset benigno, maligno e misto
os.chdir('./Kitsune')
device = Device(0)
device_dataset = dict()
if len(sys.argv) > 1:
   device = Device(int(sys.argv[1]))

device_dataset = load_dataset(device)[device.name]
clusters = load_clusters()

device_benign = device_dataset['benign_traffic']
device_benign = device_benign.sample(frac = 1) #ELIMINARE IN FASE FINALE O SETTARE A 1
device_benign = pd.concat([device_benign], ignore_index=True)
print(device_benign)
device_benign = device_benign.to_numpy()
device_benign = device_benign.astype('float32')


device_malign = pd.concat([value for key, value in device_dataset.items() if key not in ('benign_traffic')], ignore_index=True)
device_malign = device_malign.sample(frac = 1) #ELIMINARE IN FASE FINALE O SETTARE A 1
device_malign = device_malign.to_numpy().astype('float32')
device_benign = np.concatenate([device_benign], axis = 0)
device_all = np.concatenate([device_benign,device_malign], axis = 0)



tss = TimeSeriesSplit(5)

device = device.name
for algorithm in ['Kshape','KernelKmeans', 'Kmeans']:
   for key in clusters[device][algorithm].keys():
      n_autoencoder = len(clusters[device][algorithm][key]) #NB: key è il numero di clusters su cui è stato impostato l'algoritmo, il numero reale di cluster usati dall'algoritmo spesso è minore
      iteration = 0
      for train_index, test_index in tss.split(device_benign, device_benign[:,116]):
         with tf.device('/cpu:0'):
            print("Train:", train_index, "Test:", test_index)
            train_index = train_index.astype('int32')
            test_index = test_index.astype('int32')
            training = device_benign[train_index, :116]
            training_features = training[:,:115]
            training_labels = training[:, 115]
            training_labels = training_labels.astype('int')
            testing = device_benign[test_index, : 118]
            test_features = device_all[test_index, : 115]
            test_labels = device_all[test_index, 115:118]
            test_labels = test_labels.astype('int')
                  
            Ensemble = np.empty(n_autoencoder, dtype = object)
            #Building autoencoders & output
            for i, (cluster_number, cluster_elements) in enumerate(clusters[device][algorithm][key].items()): #index, (key, value)
               n_cluster_elements = len(cluster_elements)
               Ensemble[i]= Sequential()
               Ensemble[i].add(Dense(units=n_cluster_elements,activation='relu',input_shape=(n_cluster_elements,)))
               Ensemble[i].add(Dense(units=math.floor(0.75*n_cluster_elements),activation='relu'))
               Ensemble[i].add(Dense(units=n_cluster_elements,activation='sigmoid'))
               Ensemble[i].compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

            Output= Sequential()
            Output.add(Dense(units=n_autoencoder,activation='relu',input_shape=(n_autoencoder,)))
            Output.add(Dense(units=math.floor(0.75*n_autoencoder),activation='relu'))
            Output.add(Dense(units=n_autoencoder,activation='sigmoid'))
            Output.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
            scaler1=MinMaxScaler(feature_range=(0,1))
            training_features=scaler1.fit_transform(training_features)
            
            for i, (cluster_number, cluster_elements) in enumerate(clusters[device][algorithm][key].items()):
               Ensemble[i].fit(training_features[:,cluster_elements],training_features[:,cluster_elements], epochs=1, batch_size=32)
            score=np.zeros((training_features.shape[0],n_autoencoder))
            
            for j, (cluster_number, cluster_elements) in enumerate(clusters[device][algorithm][key].items()):
               pred=Ensemble[j].predict(training_features[:,cluster_elements])
               for i in range(training_features.shape[0]):
                  score[i,j]= np.sqrt(metrics.mean_squared_error(pred[i],training_features[i,cluster_elements]))
            
            
            scaler2=MinMaxScaler(feature_range=(0,1))
            score=scaler2.fit_transform(score)
            Output.fit(score,score,epochs=1,batch_size=32)
            RMSE=np.zeros(score.shape[0])
            pred=Output.predict(score)
            for i in range(score.shape[0]):
               RMSE[i]= np.sqrt(metrics.mean_squared_error(pred[i],score[i]))


            # FASE DI TESTING TSS

            test_features=scaler1.transform(test_features)
            test_score=np.zeros((test_features.shape[0],n_autoencoder))
            for j, (cluster_number, cluster_elements) in enumerate(clusters[device][algorithm][key].items()):
               pred=Ensemble[j].predict(test_features[:,cluster_elements])
               for i in range(test_features.shape[0]):
                  test_score[i,j]= np.sqrt(metrics.mean_squared_error(pred[i],test_features[i,cluster_elements]))
            
            test_score=scaler2.transform(test_score)
            RMSE=np.zeros(test_score.shape[0])
            pred=Output.predict(test_score)
            for i in range(test_score.shape[0]):
               RMSE[i]= np.sqrt(metrics.mean_squared_error(pred[i],test_score[i]))
            StampaValori(device,algorithm,n_autoencoder, iteration)


            #FASE DI TESTING SKF on TSS
            skf = StratifiedKFold(5)
            

            skf_iteration = 0
            for train_index_skf, test_index_skf in skf.split(device_malign, device_malign[:,115]):
               device_mix = np.concatenate([testing, device_malign[test_index_skf, : 118]], axis = 0)
               test_features_skf = device_mix[ :, : 115]
               test_labels_skf = device_mix[ : , 115:118]
               test_labels_skf = test_labels_skf.astype('int')

               test_features_skf=scaler1.transform(test_features_skf)
               test_score_skf=np.zeros((test_features_skf.shape[0],n_autoencoder))
               for j, (cluster_number, cluster_elements) in enumerate(clusters[device][algorithm][key].items()):
                  pred=Ensemble[j].predict(test_features_skf[:,cluster_elements])
                  for i in range(test_features_skf.shape[0]):
                     test_score_skf[i,j]= np.sqrt(metrics.mean_squared_error(pred[i],test_features_skf[i,cluster_elements]))
               
               test_score_skf = scaler2.transform(test_score_skf)
               RMSE=np.zeros(test_score_skf.shape[0])
               pred=Output.predict(test_score_skf)
               for i in range(test_score_skf.shape[0]):
                  RMSE[i]= np.sqrt(metrics.mean_squared_error(pred[i],test_score_skf[i]))
               # print(len(test_index_skf))
               # print(len(test_labels_skf[:,1]))
               # print(len(test_labels_skf[:,0]))
               # print(len(RMSE))
               StampaValoriKFold(device,algorithm,n_autoencoder,iteration,skf_iteration)
               skf_iteration=skf_iteration+1
            
            iteration = iteration+1
