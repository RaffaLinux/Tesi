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
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
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

def StampaValori(iteration,n_autoencoder):
   dataset=pd.DataFrame({'Indice': test_index, 'Dispositivo': test_labels[:,1], 'TipologiaAttacco': test_labels[:,0],'RMSE:': RMSE})
   dataset.to_csv('Ennio'+'/'+str(n_autoencoder)+'AE/Fold'+str(iteration+1)+'.csv',index=False, sep=',')

def Clustering(dataset, algorithm = 'Kmeans', device = "Ennio_Doorbell"):
   results = pd.DataFrame(columns= dataset.columns[:115])

   dataset = pd.concat([dataset], ignore_index=True)

   subset = dataset.head(n = math.floor(len(dataset.index)*0.01))
   transposed = subset.transpose()
   transposed = transposed.to_numpy().astype('float32')


   for i in [5,10,20]:
      clustering = transposed[:116, :]
      clustering_features = clustering[:115,:]
      scaler_clustering = MinMaxScaler(feature_range = (0,1))
      clustering_features = scaler_clustering.fit_transform(clustering_features)
      if algorithm == 'Kmeans':
         k = TimeSeriesKMeans(n_clusters=i, metric="softdtw", max_iter=3, verbose=0, n_jobs = 12, max_iter_barycenter = 6)
      elif algorithm == 'Kshape':
         k = KShape(n_clusters=i,max_iter= 3)
      elif algorithm == 'KernelKmeans':
         k = KernelKMeans(n_clusters = i, max_iter = 3, n_jobs = 12)
      else:
         return
      k.fit(clustering_features)
      results.loc[len(results)] = k.labels_.tolist()
   
   #results = results.transpose()
   if not os.path.isdir('./Clustering/'+device):
      os.makedirs('./Clustering/'+device)
   print(results)
   results.to_csv('./Clustering/'+device+'/'+algorithm+'.csv', index = False, sep = ',')
   
   return results




#Lettura del dataset dai csv da far diventare una funzione
os.chdir('./Kitsune-Testa')
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
   dataset[device][attack]['Attack'],dataset[device][attack]['Device'] = [Attack[attack].value,Device[device].value]
   progress += 1
   bar.update(progress)
bar.finish()



#Conversione dataset Ennio da Dataframe a numpy, creazione dataset benigno, maligno e misto
ennio_dataset = dataset['Ennio_Doorbell']
ennio_benign = ennio_dataset['benign_traffic']

Clustering(dataset = ennio_benign,algorithm = 'Kmeans', device = 'Ennio_Doorbell')

# ennio_benign = pd.concat([ennio_benign], ignore_index=True)

# ennio_benign = ennio_benign.sample(frac = 1) #ELIMINARE IN FASE FINALE
# ennio_benign = ennio_benign.to_numpy()
# ennio_benign = ennio_benign.astype('float32')


# ennio_malign = pd.concat([value for key, value in ennio_dataset.items() if key not in ('benign_traffic')], ignore_index=True)
# ennio_malign = ennio_malign.sample(frac = 0.25) #ELIMINARE IN FASE FINALE
# ennio_malign = ennio_malign.to_numpy().astype('float32')
# ennio_benign = np.concatenate([ennio_benign], axis = 0)
# ennio_all = np.concatenate([ennio_benign,ennio_malign], axis = 0)
# np.random.shuffle(ennio_all)
# np.random.seed()



# skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
# for n_autoencoder in (5,10,15,20):
#    n_autoencoder1 = 115%n_autoencoder
#    n_features2 = math.floor(115/n_autoencoder)
#    n_features1 = n_features2+1
#    n_autoencoder2 = n_autoencoder-n_autoencoder1
#    iteration = 0
#    for train_index, test_index in skf.split(ennio_all, ennio_all[:,116]):
#       with tf.device('/cpu:0'):
#          print("Train:", train_index, "Test:", test_index)
#          train_index = train_index.astype('int32')
#          test_index = test_index.astype('int32')
#          training = ennio_all[train_index, :116]
#          training = training[(training[:,115] == 0)]
#          training_features = training[:,:115]
#          training_labels = training[:, 115]
#          training_labels = training_labels.astype('int')
#          test_features = ennio_all[test_index, : 115]
#          test_labels = ennio_all[test_index, 115:117]
#          test_labels = test_labels.astype('int')
               
#          Ensemble1 = np.empty(n_autoencoder1, dtype = object)
#          Ensemble2 = np.empty(n_autoencoder2, dtype = object)
#          #Building autoencoders & output

#          for i in range(n_autoencoder1):
#             Ensemble1[i]= Sequential()
#             Ensemble1[i].add(Dense(units=n_features1,activation='relu',input_shape=(n_features1,)))
#             Ensemble1[i].add(Dense(units=math.floor(0.75*n_features1),activation='relu'))
#             Ensemble1[i].add(Dense(units=n_features1,activation='sigmoid'))
#             Ensemble1[i].compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
#          for i in range(n_autoencoder2):
#             Ensemble2[i]= Sequential()
#             Ensemble2[i].add(Dense(units=n_features2,activation='relu',input_shape=(n_features2,)))
#             Ensemble2[i].add(Dense(units=math.floor(0.75*n_features2),activation='relu'))
#             Ensemble2[i].add(Dense(units=n_features2,activation='sigmoid'))
#             Ensemble2[i].compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
#          Output= Sequential()
#          Output.add(Dense(units=n_autoencoder,activation='relu',input_shape=(n_autoencoder,)))
#          Output.add(Dense(units=math.floor(0.75*n_autoencoder),activation='relu'))
#          Output.add(Dense(units=n_autoencoder,activation='sigmoid'))
#          Output.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
#          scaler1=MinMaxScaler(feature_range=(0,1))
#          training_features=scaler1.fit_transform(training_features)
#          for i in range(n_autoencoder1):
#             Ensemble1[i].fit(training_features[:,i*n_features1:(i+1)*n_features1],training_features[:,i*n_features1:(i+1)*n_features1], epochs=1, batch_size=32)
#          for i in range(n_autoencoder2):
#             Ensemble2[i].fit(training_features[:,n_autoencoder1*n_features1+i*n_features2:n_autoencoder1*n_features1+(i+1)*n_features2],training_features[:,n_autoencoder1*n_features1+i*n_features2:n_autoencoder1*n_features1+(i+1)*n_features2], epochs=1, batch_size=32)
#          score=np.zeros((training_features.shape[0],n_autoencoder))
#          for j in range(n_autoencoder1):
#             pred=Ensemble1[j].predict(training_features[:,j*n_features1:(j+1)*n_features1])
#             for i in range(training_features.shape[0]):
#                score[i,j]= np.sqrt(metrics.mean_squared_error(pred[i],training_features[i,j*n_features1:(j+1)*n_features1]))
#          for j in range(n_autoencoder2):
#             pred=Ensemble2[j].predict(training_features[:,n_autoencoder1*n_features1+j*n_features2:n_autoencoder1*n_features1+(j+1)*n_features2])
#             for i in range(training_features.shape[0]):
#                score[i,j+n_autoencoder1]= np.sqrt(metrics.mean_squared_error(pred[i],training_features[i,n_autoencoder1*n_features1+j*n_features2:n_autoencoder1*n_features1+(j+1)*n_features2]))
#          scaler2=MinMaxScaler(feature_range=(0,1))
#          score=scaler2.fit_transform(score)
#          Output.fit(score,score,epochs=1,batch_size=32)
#          RMSE=np.zeros(score.shape[0])
#          pred=Output.predict(score)
#          for i in range(score.shape[0]):
#             RMSE[i]= np.sqrt(metrics.mean_squared_error(pred[i],score[i]))
#          # FASE DI TESTING
#          test_features=scaler1.transform(test_features)
#          test_score=np.zeros((test_features.shape[0],n_autoencoder))
#          for j in range(n_autoencoder1):
#             pred=Ensemble1[j].predict(test_features[:,j*n_features1:(j+1)*n_features1])
#             for i in range(test_features.shape[0]):
#                test_score[i,j]= np.sqrt(metrics.mean_squared_error(pred[i],test_features[i,j*n_features1:(j+1)*n_features1]))
#          for j in range(n_autoencoder2):
#             pred=Ensemble2[j].predict(test_features[:,n_autoencoder1*n_features1+j*n_features2:n_autoencoder1*n_features1+(j+1)*n_features2])
#             for i in range(test_features.shape[0]):
#                test_score[i,j+n_autoencoder1]= np.sqrt(metrics.mean_squared_error(pred[i],test_features[i,n_autoencoder1*n_features1+j*n_features2:n_autoencoder1*n_features1+(j+1)*n_features2]))
#          test_score=scaler2.transform(test_score)
#          RMSE=np.zeros(test_score.shape[0])
#          pred=Output.predict(test_score)
#          for i in range(test_score.shape[0]):
#             RMSE[i]= np.sqrt(metrics.mean_squared_error(pred[i],test_score[i]))
#          StampaValori(iteration,n_autoencoder)

#          iteration=iteration+1
