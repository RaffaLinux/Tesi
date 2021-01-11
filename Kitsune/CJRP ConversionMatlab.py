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
import scipy.io

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





os.chdir('./Kitsune')
JCRP = None
for index_dev1 in range(1,2):
   dev1 = Device(index_dev1).name
#   for index_dev2 in range(1,9):
   index_dev2 = 0
   dev2 = Device(0).name
   if index_dev2 < index_dev1:
      filename = './JCRP/'+dev1+'____'+dev2+'.np'
      f = open(filename,'rb')
      JCRP = pickle.load(f)
      scipy.io.savemat('./JCRP/'+dev1+'____'+dev2+'.mat', {"RP":JCRP})


