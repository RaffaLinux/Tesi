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
import pickle

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

def StampaValori(device,algorithm, n_clusters,iteration, index, labels, RMSE):
   dataset=pd.DataFrame({'Indice': index,'Maligno': labels[:,0], 'Dispositivo': labels[:,2], 'TipologiaAttacco': labels[:,1],'RMSE:': RMSE})

   if not os.path.isdir('./SKF/'+algorithm+str(n_clusters)):
      os.makedirs('./SKF/'+algorithm+str(n_clusters))

   dataset.to_csv('./SKF/'+algorithm+str(n_clusters)+'/SKF'+str(iteration)+'.csv',index=False, sep=',')



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
   clusters_path = './FPMaXX/'

   for algorithm in ['Kmeans','Kshape','KernelKmeans']:
      clusters[algorithm] = dict()

      rows_file = open(clusters_path+'rows_'+algorithm, 'rb')
      rows = pickle.load(rows_file)
      rows_file.close()

      solutions_file = open(clusters_path+'solutions_'+algorithm,'rb')
      solutions = pickle.load(solutions_file)
      solutions_file.close()
      clusters[algorithm]['biggest_clusters_weighted'] = fpmaxx_to_clusters(rows,solutions[0])
      clusters[algorithm]['biggest_clusters_mean'] = fpmaxx_to_clusters(rows,solutions[1])
      clusters[algorithm]['best_support_weighted'] = fpmaxx_to_clusters(rows,solutions[2])
      clusters[algorithm]['best_support_mean'] = fpmaxx_to_clusters(rows,solutions[3])
      clusters[algorithm]['best_10_mean'] = fpmaxx_to_clusters(rows,solutions[4])
      clusters[algorithm]['best_10_weighted'] = fpmaxx_to_clusters(rows,solutions[5])

   return clusters

def fpmaxx_to_clusters(rows, solution):
   clusters = dict()
   for i,elem in enumerate(solution[0]):
      clusters[i] = rows[int(elem)][0]

   print(clusters)
   return clusters



#Load del dataset,conversione dataset da Dataframe a numpy, creazione dataset benigno, maligno e all
os.chdir('./Kitsune')
dataset = dict()
device_dataset = dict()
dataset = load_dataset()
all_devices_mix = DataFrame()
all_devices_benign = DataFrame()
all_devices_malign = DataFrame()
for device in Device:
   device_dataset = dataset[device.name]
   clusters = load_clusters()

   device_benign = device_dataset['benign_traffic']
   #device_benign = device_benign.sample(frac = 1) #ELIMINARE IN FASE FINALE
   device_benign = pd.concat([device_benign], ignore_index=True)
   device_benign = device_benign.iloc[2048:]

   device_malign = pd.concat([value for key, value in device_dataset.items() if key not in ('benign_traffic')], ignore_index=True)

   all_devices_malign = pd.concat([all_devices_malign, device_malign], ignore_index=True)
   all_devices_benign = pd.concat([all_devices_benign, device_benign], ignore_index=True)

all_devices_mix = pd.concat([all_devices_benign,all_devices_malign], ignore_index= True)
all_devices_mix = all_devices_mix.to_numpy().astype('float32')
np.random.shuffle(all_devices_mix)
np.random.seed()


skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

device = device.name

for algorithm in ['Kmeans', 'Kshape', 'KernelKmeans']:
   for key in clusters[algorithm].keys():

      print("\nTraining "+algorithm+" "+str(key))
      n_autoencoder = len(clusters[device][algorithm][key]) 
      
      tss_iteration = 0

      for train_index, test_index in skf.split(all_devices_mix, all_devices_mix[:,116]):
         with tf.device('/cpu:0'):
            print("Train:", train_index, "Test:", test_index)
            train_index = train_index.astype('int32')
            test_index = test_index.astype('int32')
            training = all_devices_mix[train_index, :116]
            training = training[(training[:,115] == 0)] #Selezione dataset benevolo per il training
            training_features = training[:,:115]
            training_labels = training[:, 115]
            training_labels = training_labels.astype('int')
            testing = all_devices_mix[test_index, : 118]
            test_features = all_devices_mix[test_index, : 115]
            test_labels = all_devices_mix[test_index, 115:118]
            test_labels = test_labels.astype('int')
                  
            Ensemble = np.empty(n_autoencoder, dtype = object)

            #Building autoencoders & output
            for i, (cluster_number, cluster_elements) in enumerate(clusters[device][algorithm][key].items()): 
               #index, (key, value). i = numero ordinato del cluster, cluster_number = numero assegnato dall'algoritmo (inutile), cluster_elements = lista delle features nel cluster
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
            
            #Training. [:,cluster_elements] seleziona le colonne nella lista cluster_elements
            for i, (cluster_number, cluster_elements) in enumerate(clusters[device][algorithm][key].items()):
               Ensemble[i].fit(training_features[:,cluster_elements],training_features[:,cluster_elements], epochs=1, batch_size=32)
            score=np.zeros((training_features.shape[0],n_autoencoder))
            
            #Generazione score Ensemble layer. i itera sulle entries del dataset, j sugli autoencoder/cluster
            for j, (cluster_number, cluster_elements) in enumerate(clusters[device][algorithm][key].items()):
               pred=Ensemble[j].predict(training_features[:,cluster_elements])
               for i in range(training_features.shape[0]):
                  score[i,j]= np.sqrt(metrics.mean_squared_error(pred[i],training_features[i,cluster_elements]))
            
            
            scaler2=MinMaxScaler(feature_range=(0,1))
            score=scaler2.fit_transform(score)

            #Training e generazione score Output layer
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
            
            
            StampaValori(device,algorithm,key, tss_iteration, test_index, test_labels, RMSE)
            
            tss_iteration = tss_iteration+1
