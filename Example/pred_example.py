import nibabel as nib
import numpy as np
import glob
import os
import sys
import component_cut
import cv2
from os import path
import unet
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from tensorflow import Graph
from tensorflow import Session
import pandas as pd
from keras.backend.tensorflow_backend import set_session


sys.stdout.flush()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.Session(config=config))


K.set_image_data_format('channels_last')  # TF dimension ordering in this code
size=1280


def dice_index(y_true, y_pred, smooth=1): 
    '''
    input: true label; predicted label; smooth
    output: dice coefficient
    '''
    numerator = 2.*np.sum(np.multiply(y_true, y_pred))
    denominator = np.sum(y_pred + y_true)
    return (numerator + smooth) / (denominator + smooth)

def f2(element, thresh):
    '''
    input: pixel element; thresold 
    output: 1 if element value >=threshold; 0 otherwise
    '''
    return 1 if element >= thresh else 0
f2 = np.vectorize(f2)


json = pd.read_json("probe_correspondant.json")
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
        
        


model0 = unet.get_unet()
model0.load_weights('weights_0.h5')

model1 = unet.get_unet()
model1.load_weights('weights_1.h5')

model2 = unet.get_unet()
model2.load_weights('weights_2.h5')

model3 = unet.get_unet()
model3.load_weights('weights_3.h5')

model4 = unet.get_unet()
model4.load_weights('weights_4.h5')



image1=cv2.imread('GSM707032.CEL.png') # Change the path to your cel file
image1=(image1-np.mean(image1))/np.std(image1) #standardize test image
image=np.zeros((size,size,1))
image[0:image1.shape[0],0:image1.shape[1],0]=image1[:,:,0] #reshape to 1280*1280
image_batch=[] 
image_batch.append(image) #contatenate each test image to image_batch
image_batch=np.array(image_batch)


pred_new=np.zeros((size,size))
pred=model0.predict(image_batch)
pred_new+=pred[0,:,:,0]

pred=model1.predict(image_batch)
pred_new+=pred[0,:,:,0]


pred=model2.predict(image_batch)
pred_new+=pred[0,:,:,0]

pred=model3.predict(image_batch)
pred_new+=pred[0,:,:,0]

pred=model4.predict(image_batch)
pred_new+=pred[0,:,:,0]

pred_new=pred_new/5.0
pred_new=pred_new[0:image1.shape[0],0:image1.shape[1]]


np.save('./result.npy',pred_new)


pred_new = f2(pred_new, 0.5) 

# get the name set probesets with contamination
probe_contam = set(probe_mat[pred_new==1])
probe_contam = [i for i in list(probe_contam) if i != '']   #remove ''

#get the name set of contmainated gene and non-contaminated genes
gene_contam = list(set(probe2gene[i] for i in probe_contam if str(probe2gene[i])!= 'nan'))
gene_contam = np.array(gene_contam)
np.savetxt('contam_genes.txt', gene_contam, fmt='%s')