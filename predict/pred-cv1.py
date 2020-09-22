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
from keras.backend.tensorflow_backend import set_session

sys.stdout.flush()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.Session(config=config))
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
size=1280
FILE=open('test_1_cv.dat','r')
os.system('rm -rf result')
os.system('mkdir result')


def dice_index(y_true, y_pred, smooth=1): 
    '''
    input: true label; predicted label; smooth
    output: dice coefficient
    '''
    numerator = 2.*np.sum(np.multiply(y_true, y_pred))
    denominator = np.sum(y_pred + y_true)
    return (numerator + smooth) / (denominator + smooth)

def f(element, thresh):
    return 1 if element >= thresh else 0
f = np.vectorize(f)



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


NEW=open('eva_cv1.txt','w')
for line in FILE:
    case = 'negative'
    line=line.rstrip()
    the_id=line
    image1=cv2.imread(the_id) #read each test image
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
    
    
    pred_new_save = f(pred_new, 0.5) 
    
    #save the predicted matrix into a place for future use
    if np.sum(pred_new_save)>0:
        case = 'positive'
        np.save('/media/ssd2/yananq/mace/code/pos_pred_cv1/'+the_id.split('/')[-1], pred_new)
    
    # get y_true
    label = np.zeros((image1.shape[0],image1.shape[1])) 
    img_name = the_id.split('/')[-1]
    label_path = '../../data/label/' + img_name
    if path.exists(label_path):
        label = cv2.imread(label_path)
        label = label[:,:,0]/255
    
    #calculate dice index
    diff = -1
    diff = dice_index(label, pred_new)
    print(diff, flush=True)
    NEW.write(('%s\t%.9f\t%s\n') % (the_id,diff,case))
