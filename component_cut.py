#!/usr/bin/env python

#PathDicom = "/home/gyuanfan/2016/DM/data/pilot_images/"
#from gzip import GzipFile
import cv2
import numpy as np
import skimage
from skimage import measure
import copy
import random
#import logging
#import time

import os


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return(cmin,cmax,rmin,rmax)

def read_img_pix(the_head_matrix):
    
    (x,y,z)=the_head_matrix.shape
    return_map=np.zeros((x,y,z))
#    print(x,y,z)

    the_mean=np.mean(the_head_matrix)
    i=0
    while (i<z):

        kernel = np.ones((3,3),np.uint16)
        dilation = cv2.dilate(the_head_matrix[:,:,i],kernel,iterations = 1)
        dilation[dilation<the_mean]=0
        dilation[dilation>=the_mean]=255
        dilation=dilation.astype('uint8')

    
        labels = measure.label(dilation)
        vals, counts = np.unique(labels, return_counts=True)

        max_val=0
        max_i=0
        for iii in vals:
            img_seg=copy.copy(the_head_matrix[:,:,i])
            img_seg[labels>iii]=0
            img_seg[labels<iii]=0
            if (img_seg.sum()>max_val):
                max_val=img_seg.sum()
                max_i=iii

        dilation[labels>max_i]=0
        dilation[labels<max_i]=0
        return_map[:,:,i]=dilation
        i=i+1
    return(return_map)

