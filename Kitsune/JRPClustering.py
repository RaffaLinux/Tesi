import glob
import pandas as pd
import sys
import math
import os
import datetime
import progressbar
import numpy as np
import time
import pickle
#from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from sklearn.preprocessing import MinMaxScaler
from tslearn.preprocessing import TimeSeriesScalerMinMax
from matplotlib import pyplot as plt
from matplotlib.pyplot import legend
from enum import Enum
from pyts.multivariate.image import JointRecurrencePlot
from pyts.image import RecurrencePlot
from matplotlib.colors import SymLogNorm
from pyrqa.computation import JRQAComputation
from pyrqa.metric import MaximumMetric, TaxicabMetric
from pyrqa.settings import JointSettings
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation

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

def generate_joint_recurrence_plot(dataset,device):
   dataset = pd.concat([dataset], ignore_index=True)
   subset = dataset.head(n= 2048)
   subset = subset.to_numpy().astype('float32')
   ftrs = subset[:, :116]
   features = ftrs[:,:115]
   scaler = MinMaxScaler(feature_range = (0,1))
   features = scaler.fit_transform(features)
   features = np.transpose(features)
   print(features.shape)
   features = features.reshape((1,115,2048))
   print(features.shape)
   jrp = JointRecurrencePlot(threshold='distance')
   features_jrp = jrp.fit_transform(features)
   
   filename = './JRP/'+device
   outfile = open(filename,'wb')
   pickle.dump(features_jrp[0],outfile)
   outfile.close()

   plt.figure(figsize=(5, 5))
   plt.imshow(features_jrp[0],norm=SymLogNorm(linthresh=1e-3,vmin=0,vmax=1))
   plt.gca().invert_yaxis()
   plt.title(device.replace('_',' '))
   plt.colorbar(shrink = .7)
   plt.tight_layout()
   plt.savefig('./Graphs/Joint Recurrence Plot/'+device+'.pdf')

   return


os.chdir('./Kitsune')
device = Device(0)
dataset = dict()

if len(sys.argv) > 1:
   device = Device(int(sys.argv[1]))

dataset = load_dataset()

dataset_benign = dict()

for dev in Device:
   dataset_benign[dev.name] = dataset[dev.name]['benign_traffic']
   dataset_benign[dev.name] = pd.concat([dataset_benign[dev.name]], ignore_index=True)
   dataset_benign[dev.name] = dataset_benign[dev.name].head(n = 2048)
   dataset_benign[dev.name] = dataset_benign[dev.name].to_numpy().astype('float32')
   dataset_benign[dev.name] = dataset_benign[dev.name][:,:116]
   dataset_benign[dev.name] = dataset_benign[dev.name][:,:115]
   scaler = MinMaxScaler(feature_range = (0,1))
   dataset_benign[dev.name] = scaler.fit_transform(dataset_benign[dev.name])
   print(dataset_benign[dev.name])

for i, dev1 in enumerate(Device):
   time_serieses_1 = []
   for k in range(115):
      time_serieses_1.append(TimeSeries(dataset_benign[dev1.name][:,k]))
   settings_1 = Settings(time_serieses_1)

   for j,dev2 in enumerate(Device):
      if j > i:
            time_serieses_2 = []
            for k in range(115):
               time_serieses_2.append(TimeSeries(dataset_benign[dev2.name][:,k]))
            settings_2 = Settings(time_serieses_2)
            joint_settings = JointSettings(settings_1, settings_2)
            computation = JRQAComputation.create(joint_settings, verbose=True)
            result = computation.run()
            print(result)
