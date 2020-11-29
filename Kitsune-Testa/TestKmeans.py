import glob
import pandas as pd
import math
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from enum import Enum

class Device(Enum):
   Danmini_Doorbell = 1
   Ecobee_Thermostat = 2
   Ennio_Doorbell = 3
   Philips_B120N10_Baby_Monitor = 4
   Provision_PT_737E_Security_Camera = 5
   Provision_PT_838_Security_Camera = 6
   Samsung_SNH_1011_N_Webcam = 7
   SimpleHome_XCS7_1002_WHT_Security_Camera = 8
   SimpleHome_XCS7_1003_WHT_Security_Camera = 9

class Attack(Enum):
   benign_traffic = 1
   gafgyt_combo = 2
   gafgyt_junk = 3
   gafgyt_scan = 4
   gafgyt_tcp = 5
   gafgyt_udp = 6
   mirai_ack = 7
   mirai_scan = 8
   mirai_syn = 9
   mirai_udp = 10
   mirai_udpplain = 11




#Lettura del dataset dai csv da far diventare una funzione
os.chdir('./Kitsune-Testa')
dataset = dict()
dataset_path = "./Dataset/"
csv_paths = glob.glob(dataset_path+'**/*.csv', recursive = True)
for csv_path in csv_paths:
   print(csv_path)
   attack = ''
   device = csv_path.split('\\')[1]
   if device not in dataset:
      dataset[device] = {}
   if( len(csv_path.split('\\')) == 4):
      attack = csv_path.split('\\')[2]+'_'+csv_path.split('\\')[3]
   else:
      attack = csv_path.split('\\')[2]
   attack = attack.replace(".csv","")
   print(attack)
   dataset[device][attack] = pd.read_csv(csv_path, delimiter = ',')
   dataset[device][attack]['Attack'],dataset[device][attack]['Device'] = [Attack[attack].value,Device[device].value]

#Conversione dataset Danmini da Dataframe a numpy, creazione dataset benigno, maligno e misto
danmini_dataset = dataset['Danmini_Doorbell']
danmini_benign = danmini_dataset['benign_traffic']
danmini_benign = pd.concat([danmini_benign], ignore_index=True)
print(danmini_benign)
danmini_benign = danmini_benign.to_numpy()
dabnubu_benign = danmini_benign.astype('float32')
danmini_malign = pd.concat([value for key, value in danmini_dataset.items() if key not in ('benign_traffic')], ignore_index=True).to_numpy().astype('float32')
danmini_all = np.concatenate([danmini_benign,danmini_malign], axis = 0)
np.random.shuffle(danmini_all)
np.random.seed()


skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
for n_autoencoder in (5,10,15,20):
   n_autoencoder1 = 115%n_autoencoder
   n_features2 = math.floor(115/n_autoencoder)
   n_features1 = n_features2+1
   n_autoencoder2 = n_autoencoder-n_autoencoder1
   iteration = 0
   for train_index, test_index in skf.split(danmini_all, danmini_all[:,115]):
      with tf.device('/cpu:0'):
         print("Train:", train_index, "Test:", test_index)
         train_index = train_index.astype('int32')
         test_index = test_index.astype('int32')
         training = danmini_all[train_index, :116]
         training = training[(training[:,115] == 1)]
         training_features = training[:,:115]
         training_labels = training[:, 115]
         training_labels = training_labels.astype('int')
         test_features = danmini_all[test_index, : 115]
         test_labels = danmini_all[test_index, 115:117]
         test_labels = test_labels.astype('int')
               
         Ensemble1 = np.empty(n_autoencoder1, dtype = object)
         Ensemble2 = np.empty(n_autoencoder2, dtype = object)
         #Building autoencoders & output
         for i in range(n_autoencoder1):
            Ensemble1[i] = Sequential()
            Ensemble1[i].add(Dense(units=n_features1, activation = 'relu', input_shape = (n_features1,)))
            Ensemble1[i].add(Dense(units=math.floor(0.75*n_features1), activation = 'relu'))
            Ensemble1[i].add(Dense(units = n_features1, activation = 'sigmoid'))
            Ensemble1[i].compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
               
         for i in range(n_autoencoder2):
            Ensemble2[i] = Sequential()
            Ensemble2[i].add(Dense(units = n_features2, activation = 'relu', input_shape = (n_features2,)))
            Ensemble2[i].add(Dense(units = math.floor(0.75*n_features2), activation = 'relu'))
            Ensemble2[i].add(Dense(units = n_features2, activation = 'sigmoid'))
            Ensemble2[i].compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

         Output = Sequential()
         Output.add(Dense(units = n_autoencoder, activation = 'relu', input_shape = (n_autoencoder,)))
         Output.add(Dense(units = math.floor(0.75/n_autoencoder), activation = 'relu'))
         Output.add(Dense(units = n_autoencoder, activation = 'sigmoid'))
         Output.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

         #Fitting autoencoders
         scaler1 = MinMaxScaler(feature_range = (0,1))
         training_features = scaler1.fit_transform(training_features)
         print("Fitting n_autoencoder1")

         for i in range(n_autoencoder1):
            Ensemble1[i].fit(training_features[:,i*n_features1 : (i+1)*n_features1], training_features[:, i*n_features1 : (i+1)*n_features1], epochs = 1, batch_size = 32)

         print("Fitting n_autoencoder2")

         for i in range(n_autoencoder2):
            Ensemble2[i].fit(training_features[:,n_autoencoder1*n_features1+i*n_features2:n_autoencoder1*n_features1+(i+1)*n_features2],training_features[:,n_autoencoder1*n_features1+i*n_features2:n_autoencoder1*n_features1+(i+1)*n_features2], epochs=1, batch_size=32)

         score=np.zeros((training_features.shape[0],n_autoencoder))
         print("Pred n_autoencoder1")

         for j in range(n_autoencoder1):
            pred = Ensemble1[j].predict(training_features[:,j*n_features1 : (j+1)*n_features1])
                  
            for i in range(training_features.shape[0]):
               score[i,j] = np.sqrt(metrics.mean_squared_error(pred[i], training_features[i, j*n_features1 : (j+1)*n_features1]))

         print("Pred n_autoencoder2")

         for j in range(n_autoencoder2):
            pred = Ensemble2[j].predict(training_features[: , n_autoencoder1*n_features1 + j*n_features2 : n_autoencoder1*n_features1 + (j+1)*n_features2])

            for i in range(training_features.shape[0]):
               score[i,j+n_autoencoder1] = np.sqrt(metrics.mean_squared_error(pred[i], training_features[i, n_autoencoder1*n_features1 + j*n_features2: n_autoencoder1*n_features1+(j+1)*n_features2]))
               

         scaler2 = MinMaxScaler(feature_range = (0,1))
         score = scaler2.fit_transform(score)
         print("Output fit")
#Aggiustare gli spazi di questa roba qua sotto
         Output.fit(score,score, epochs = 1, batch_size = 32)
         RMSE = np.zeros(score.shape[0])
         print("Output pred")
         pred = Output.predict(score)
         for i in range(score.shape[0]):
            RMSE[i] = np.sqrt(metrics.mean_squared_error(pred[i], score[i]))
         test_features=scaler1.transform(test_features)
         test_score=np.zeros((test_features.shape[0],n_autoencoder))
         for j in range(n_autoencoder1):
            pred=Ensemble1[j].predict(test_features[:,j*n_features1:(j+1)*n_features1])
            for i in range(test_features.shape[0]):
               test_score[i,j]= np.sqrt(metrics.mean_squared_error(pred[i],test_features[i,j*n_features1:(j+1)*n_features1]))
         for j in range(n_autoencoder2):
            pred=Ensemble2[j].predict(test_features[:,n_autoencoder1*n_features1+j*n_features2:n_autoencoder1*n_features1+(j+1)*n_features2])
            for i in range(test_features.shape[0]):
               test_score[i,j+n_autoencoder1]= np.sqrt(metrics.mean_squared_error(pred[i],test_features[i,n_autoencoder1*n_features1+j*n_features2:n_autoencoder1*n_features1+(j+1)*n_features2]))
         test_score=scaler2.transform(test_score)
         RMSE=np.zeros(test_score.shape[0])
         pred=Output.predict(test_score)
         for i in range(test_score.shape[0]):
            RMSE[i]= np.sqrt(metrics.mean_squared_error(pred[i],test_score[i]))
