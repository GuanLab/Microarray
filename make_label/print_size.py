import cv2
import numpy as np
import os
import glob


all_label=glob.glob('split_processed/*')

for the_label in all_label:
    the_name=the_label.split('/')
#    print(('raw_image/'+the_name[-1]))
    img=cv2.imread('raw_image/mace_full/'+the_name[-1])
    (aaa,bbb)=(img.shape[0],img.shape[1])
    print(aaa,bbb)
