# author: Yanan Qin 

import numpy as np
import pandas as pd
import seaborn as sns
import json
import os
import os.path
import shutil
import random
import nibabel as nib
import numpy as np
import glob
import sys
import component_cut
import cv2
import unet
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from tensorflow import Graph
from os import path
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))



# The recover_intensity.py save the recovered sample as an 1164*1164 matrix in a txt file. 
# This piece of code is to save the recover sample into a probe: intensity match style



# read HG-U133_Plus_2.cdf file and extract information
index2loc = dict()

def process_line(raw):
    '''
    input: raw line
    output: X, Y location; index of probes
    '''
    line = raw.strip()
    parts = line.split('\t')
    X = int(parts[0].split('=')[-1])
    Y = int(parts[1])
    index = parts[11]
    return X, Y, index

# read the CDF file and read out probe information
# this piece of code is modified from extract_cdf.py, whose author is Xianghao Chen
concern = False
start = False
with open("HG-U133_Plus_2.cdf",'r') as f:
        for line in f:
            if line[:5] == "[Unit" and line[-9:-3] == "_Block":
                concern = True
                continue
            if not concern:
                continue
            if line[:12] == "CellHeader=X":
                start = True
                continue
            if start:
                if line[:4] == "Cell":
                    X, Y, index = process_line(line)
                    index2loc[index] = (X, Y)
                else:
                    concern = False
                    start = False


print(len(index2loc.keys())) # number of probes that are available in the cdf file, i.e. have corresponding probesets
# = 1208516


# read each itermediate output and save as final output
for i in os.listdir('/media/ssd/yananq/mace/recovered_contam_intensities/'):
    NEW=open('/media/ssd/yananq/mace/recovered_contam_intensities_map_final/'+i,'w')
    sample = pd.read_table('/media/ssd/yananq/mace/recovered_contam_intensities/'+i, ' ', header =None)
    sample = np.asmatrix(sample, dtype=float)
    for index, loc in index2loc.items():
        X = loc[0]
        Y = loc[1]
        intensity = sample[X, Y]
        NEW.write(('%s\t%f\n') % (index,intensity))
