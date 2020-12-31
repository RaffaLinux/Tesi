import glob
import pandas as pd
import sys
import math
import os
import datetime
import progressbar
import numpy as np
import time
#from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from sklearn.preprocessing import MinMaxScaler
from tslearn.preprocessing import TimeSeriesScalerMinMax
from matplotlib import pyplot as plt
from matplotlib.pyplot import legend
from enum import Enum
from pyts.multivariate.image import JointRecurrencePlot
from pyts.image import RecurrencePlot


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

def generate_joint_recurrence_plot(dataset):
   stacked_datasets = np.stack(dataset.values(), axis=0)
   ftrs = stacked_datasets[:,:, :116]
   features = ftrs[:,:,:115]
   print(features.shape)
   jrp = JointRecurrencePlot()
   features_jrp = jrp.fit_transform(features)
   plt.figure(figsize=(5, 5))
   plt.imshow(features_jrp[0], cmap='binary', origin='lower')
   plt.title('Joint Recurrence Plot', fontsize=18)
   plt.tight_layout()
   plt.savefig('./Graphs/Joint Recurrence Plot/JointRecurrencePlot.pdf')

   return


os.chdir('./Kitsune')
device_dataset = dict()


dataset = load_dataset()
benign_datasets = dict()
scaler = MinMaxScaler(feature_range = (0,1))

for dev in Device:
   subset = dataset[dev.name]['benign_traffic']
   subset = pd.concat([subset], ignore_index=True)
   subset = subset.head(n = 2048)
   subset = subset.to_numpy().astype('float32')
   subset = scaler.fit_transform(subset)
   benign_datasets[dev.name] = subset



generate_joint_recurrence_plot(benign_datasets)
