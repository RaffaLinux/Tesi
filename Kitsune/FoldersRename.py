import os
import sys
from enum import Enum
import glob

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

os.chdir('./Kitsune/SKF/Hybrid/Randoms')

paths = glob.glob('.\\**\\', recursive = True)
print(paths)
for dev in Device:
    for path in paths:
        try:
            os.rename(path, path.replace(dev.name, str(dev.value)))
        except:
            print("Skipping path "+ path)