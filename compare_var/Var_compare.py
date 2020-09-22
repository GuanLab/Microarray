# author: Yanan Qin
import numpy as np
import pandas as pd
import seaborn as sns
import json
import os
import shutil
import random
from os import path
import nibabel as nib
import glob
import sys
import cv2
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from tensorflow import Graph
from tensorflow import Session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
size=1280
os.system('rm -rf result')
os.system('mkdir result')

def f2(element, thresh):
    '''
    input: pixel element; thresold 
    output: 1 if element value >=threshold; 0 otherwise
    '''
    return 1 if element >= thresh else 0
f2 = np.vectorize(f2)


avg_sample = pd.read_csv('/media/ssd/yananq/mace/avg_neg_sample.csv')
avg_sample.index = avg_sample['Unnamed: 0']
avg_sample = avg_sample.drop(columns=['Unnamed: 0'])['0']



json = pd.read_json("/media/ssd2/yananq/mace/code/baseline/probe_correspondant.json")
map_dict  = json[0:54675]
probe2gene = map_dict.to_dict()['probe2gene']
#building a matrix with probe set ids's at their corresponding location
probe_mat = np.empty((1164, 1164), dtype=np.dtype('U100'))
for index, row in map_dict.iterrows():
    probe_set = row.name
    x_loc = row['dict'][0]
    y_loc = row['dict'][1]
    for i in range(len(x_loc)):
        probe_mat[x_loc[i],y_loc[i]] = probe_set



# map the cel file name to the location of corresponding rma file
cel_rma_path = dict()
directory = r'/media/ssd/yananq/mace/bg'
for filename in os.listdir(directory):
    filename = filename.strip()
    for cel in os.listdir(directory +'/'+ filename):
        pathh = '/media/ssd/yananq/mace/cel_expression/' +filename +'.txt'
        cel_rma_path[cel] = pathh
    
    
    
NEW=open('var_contam_vs_noncontam.txt','w')        
cnt = 0      
for filename in os.listdir('/media/ssd2/yananq/mace/code/pos_pred/'):# for each sample predicted as positive
    pred_new  = np.load('/media/ssd2/yananq/mace/code/pos_pred/'+filename)
    pred_new = f2(pred_new, 0.5) 
    if True: 
        cel = '.'.join(filename.strip().split('.')[:-2])
        print(cel)  
        try:
            # get the probe expression of the sample and do correction
            probe_expression = pd.read_table(cel_rma_path[cel])
            probe_expression.index = probe_expression['Unnamed: 0']
            try:
                probe_expression = probe_expression.drop(columns=['Unnamed: 0'])[cel[:-4]]/avg_sample
            except:
                probe_expression = probe_expression.drop(columns=['Unnamed: 0'])[cel]/avg_sample

            # get the name set probes with contamination
            probe_contam = set(probe_mat[pred_new==1])
            probe_contam = [i for i in list(probe_contam) if i != '']   #remove ''

            #get the name set of probes with no contam
            probe_noncontam = probe2gene.keys()-probe_contam

            #get the name set of contmainated gene and non-contaminated genes
            gene_contam = set(probe2gene[i] for i in probe_contam if str(probe2gene[i])!= 'nan')
            gene_noncontam = set(probe2gene[i] for i in probe_noncontam if str(probe2gene[i])!= 'nan')
            gene_noncontam = gene_noncontam - gene_noncontam.intersection(gene_contam)

            probe_expression = pd.DataFrame(probe_expression)

            probe_expression['probe'] = probe_expression.index
            probe_expression['gene'] = probe_expression['probe'].map(probe2gene)
            probe_expression = probe_expression.dropna().set_index(['gene', 'probe'])

            # calculate variance intra-gene expression
            gene_var = probe_expression.groupby(['gene'], as_index=True).agg(np.var).dropna()

            gene_contam_var = gene_var[gene_var.index.isin(gene_contam)].mean()[0]
            gene_noncontam_var = gene_var[gene_var.index.isin(gene_noncontam)].mean()[0]


            print((gene_contam_var, gene_noncontam_var))
            NEW.write(('%s\t%s\t%.9f\n') % (cel, gene_contam_var, gene_noncontam_var))
            cnt +=1
        except:
            print('pass')
            pass
    
print (cnt) #used as the denominater when calculating % of the images,  {s.d.contaminated} showed higher standard error overall than the {s.d.uncontaminated} group
# result = 1163