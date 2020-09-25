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
from keras.models import load_model
from tensorflow import Graph
from os import path


def get_lists_Xmeanijkn_and_Xijkn(ref, probe_set, pred_new, map_dict, to_recover):
    '''
    input: reference matrix; probeset name; predicted mask; a data frame obtained from json file; matrix to be recovered
    output: list(length=n) of Xreferenceijkn and list(length=n) of Xijkn (defined in the paper)
    '''
    
    #get the X/Y location of probes from the same probe set
    locX = map_dict.loc[probe_set].dict[0]
    locY = map_dict.loc[probe_set].dict[1]

    X_meanijkn  = list()
    X_ijkn  = list()
    for i in range(len(locX)): 
        if pred_new[locX[i],locY[i]] == 1: # if the probe is contaminated, pass
            pass
        else:# if not, save values
            X_meanijkn.append(ref[locX[i], locY[i]])
            X_ijkn.append(to_recover[locX[i], locY[i]])
    return (X_meanijkn, X_ijkn)


def f2(element, thresh):
    '''
    input: pixel element; thresold 
    output: 1 if element value >=threshold; 0 otherwise
    '''
    return 1 if element >= thresh else 0
f2 = np.vectorize(f2)



json = pd.read_json("probe_correspondant.json")
map_dict  = json[0:54675]

#building a matrix with probe set ids's at their corresponding location
probe_mat = np.empty((1164, 1164), dtype=np.dtype('U100'))
for index, row in map_dict.iterrows():
    probe_set = row.name
    x_loc = row['dict'][0]
    y_loc = row['dict'][1]
    for i in range(len(x_loc)):
        probe_mat[x_loc[i],y_loc[i]] = probe_set
probe_mat = np.asmatrix(probe_mat)


#load reference matrix and save as matrix for consistance
ref = pd.read_table('reference.txt', ' ', header =None)
ref =np.asmatrix(ref.values)



contam_probe_cnt = 0 #number of probes we found contaminated and tried to recover
fail_recover_probe_cnt = 0 # the number of proves we could not recover


#read prediction mask
pred_new = np.load('result.npy')
pred_new = f2(pred_new, 0.5) 
pred_new = np.asmatrix(pred_new)

#read probe intensities
to_recover = pd.read_table('to_recover.txt', ' ', header =None)
to_recover = np.asmatrix(to_recover, dtype=float)


# read each cell in the mask
for row in range(pred_new.shape[0]):
    for col in range(pred_new.shape[1]):
        if pred_new[row,col] ==1: # if the probe at [row, col] is contaminated
            probe_set = probe_mat[row, col] # get the corresponding probeset 
            if probe_set!='': # if have probeset correspondance, we try to recover
                # try to recover:
                contam_probe_cnt += 1
                X_meanijkc = ref[row, col]
                (list_X_meanijkn, list_X_ijkn) = get_lists_Xmeanijkn_and_Xijkn(ref, probe_set, pred_new, map_dict, to_recover)
                if len(list_X_meanijkn)>0: # can recover
                    bar_X_meanijkn = sum(list_X_meanijkn)/len(list_X_meanijkn)
                    bar_X_ijkn = sum(list_X_ijkn)/len(list_X_ijkn)
                    to_recover[row, col] = (X_meanijkc*bar_X_ijkn)/bar_X_meanijkn
                else: # list is empty, cannot recover
                    fail_recover_probe_cnt +=1
                    to_recover[row, col] =np.NaN
            else:# if have no corresponding probe set, do not have to recover
                pass
# np.savetxt('recovered.txt',to_recover,fmt='%.f') # save the matrix into a txt file (intermediate result)
                   
                   
                   

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


NEW=open('recovered.txt','w')
sample = np.asmatrix(to_recover, dtype=float)
for index, loc in index2loc.items():
    X = loc[0]
    Y = loc[1]
    intensity = sample[X, Y]
    NEW.write(('%s\t%f\n') % (index,intensity))
