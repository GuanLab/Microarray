from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import keras
import cv2
import sys
from keras.backend.tensorflow_backend import set_session
import unet
import random
import GS_split

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))



K.set_image_data_format('channels_last')  # TF dimension ordering in this code
size= 1280
batch_size=2
model = unet.get_unet()


(train_line,test_line)=GS_split.GS_split('train_4_cv.dat',size)  
# to split train and train-validation set

def generate_data(train_line, batch_size):
    """Replaces Keras' native ImageDataGenerator."""
    # a generator, return
    # image_batch: list containing original images for one batch
    # label_batch_0: list containing zero matrix if negative/labeled image if positive for one batch
    i = 0
    while True:
        image_batch = []
        label_batch = []
        for b in range(batch_size):
            if i == len(train_line): #if reach the end of the whole set, one epoch ends; reshuffle and start again
                i = 0
                random.shuffle(train_line)
            sample = train_line[i]
            i += 1
            t=sample.split('\t')   # t[0] image; t[1] neg/labeled image if positive
            image=np.zeros((size,size,1)) #1280*1280*1 one channel
            image1 = cv2.imread(t[0]) 
            image1=(image1-np.mean(image1))/np.std(image1) #standardize image pixels

            (aaa,bbb)=image1.shape[0:2]
            image[0:aaa,0:bbb,0]=image1[:,:,0]   #augment image to 1280*1280 (zero for rest)

            image_batch.append(image)

            
            
            label=np.zeros((size,size,1)) 

            if (t[1] == 'neg'):
                pass
            else: 
                a=cv2.imread(t[1])
                label[0:aaa,0:bbb,0]=a[:,:,0]/255.0

            label_batch.append(label)
        label_batch=np.array(label_batch)
        image_batch=np.array(image_batch)
        
        yield (image_batch, label_batch)

        
callbacks = [
    keras.callbacks.TensorBoard(log_dir='./',
    histogram_freq=0, write_graph=True, write_images=False),
    keras.callbacks.ModelCheckpoint(os.path.join('./', "weights_cv4.h5"),
    verbose=0, save_weights_only=True)#,monitor='val_loss')
    ]

print("###########################################################################################################")
print(len(train_line), batch_size, len(test_line))

model.fit_generator(
    generate_data(train_line, batch_size), #generate training data
    steps_per_epoch=int(len(train_line) // batch_size), nb_epoch=5,validation_data=generate_data(test_line,batch_size),validation_steps=100,callbacks=callbacks)

