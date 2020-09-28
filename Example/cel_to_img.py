#!/usr/bin/env python3
import math
import os
import sys
sys.path.append('build/lib.linux-x86_64-3.8')
import random
import numpy as np
import subprocess as sp
import cv2
#import picpac
import cMace

N_SAMPLES = 100
N_PROBE_SETS = cMace.PS
ROWS = cMace.HEIGHT
COLS = cMace.WIDTH

path='./GSM707032.CEL'  # change the path to yours
print(path)
t=path.split('/')
name='./'+t[-1]+'.png'

cel = cMace.load_cel(path)
if cel.shape[0] == 0:
    sp.call("cp '%s' '%s.bak'" % (path, path), shell=True)
    sp.call("./apt-cel-convert -i -f xda '%s'" % path, shell=True)
    cel = cMace.load_cel(path)

x=int(cel.shape[0])
cel=cv2.resize(cel,(x,x))
cel=((cel-np.median(cel))/np.std(cel)+0.5)*160
cv2.imwrite(name, np.clip(cel , 0, 255))
