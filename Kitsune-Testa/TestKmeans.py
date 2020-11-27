import glob
import pandas as pd
import math
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import f1_score

os.chdir('./Kitsune-Testa')
dataset = dict()
dataset_path = "./Dataset/"
csv_paths = glob.glob(dataset_path+'**/*.csv', recursive = True)
for csv_path in csv_paths:
   print(csv_path)
   subkey = ''
   if csv_path.split('\\')[1] not in dataset:
      dataset[csv_path.split('\\')[1]] = {}
   if( len(csv_path.split('\\')) == 4):
      subkey = csv_path.split('\\')[2]+'_'+csv_path.split('\\')[3]
   else:
      subkey = csv_path.split('\\')[2]
   subkey = subkey.replace(".csv","")
   print(subkey)
   dataset[csv_path.split('\\')[1]][subkey] = pd.read_csv(csv_path, delimiter = ',')

print(dataset)
print(dataset.keys())
danmini_dataset = dataset['Danmini_Doorbell']
print(danmini_dataset.keys())
danmini_benign =danmini_dataset['benign_traffic']
danmini_benign = pd.concat([danmini_benign], ignore_index=True)
danmini_benign = danmini_benign.to_numpy()
dabnubu_benign = danmini_benign.astype('float32')
danmini_malign = pd.concat([value for key, value in danmini_dataset.items() if key not in ('benign_traffic')], ignore_index=True).to_numpy().astype('float32')
danmini_all = np.concatenate([danmini_benign,danmini_malign], axis = 0)
np.random.shuffle(danmini_all)
np.random.seed()


for n_autoencoder in (5,10,15,20):
    n_autoencoder1 = 115%n_autoencoder
    n_features2 = math.floor(115/n_autoencoder)
    n_features1 = n_features2+1
    n_autoencoder2 = n_autoencoder-n_autoencoder1
iteration = 0
