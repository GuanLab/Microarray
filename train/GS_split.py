#!/usr/bin/env python

import random
import cv2
import numpy as np
from datetime import datetime
import re

def GS_split (FILE,size): #size = 1280
    random.seed(datetime.now())
    TRAIN_LIST=open(FILE,'r')

    ## count pos and neg
    pos_count=0
    neg_count=0
    
    all_train_neg_line=[]
    all_train_pos_line=[]
    for line in TRAIN_LIST:
        line=line.rstrip()
        table=line.split('\t')
        tmp=table[1].split('/')
        if (table[1] == 'neg'):
            all_train_neg_line.append(line)

        else:
            all_train_pos_line.append(line)

    random.shuffle(all_train_neg_line)
    random.shuffle(all_train_pos_line)

    all_train_train_line=[]
    all_train_test_line=[]

########## partition negative #############
    cut1=int(len(all_train_neg_line)/2)
    cut2=int(len(all_train_neg_line))

    i=0
    while (i<cut1):
        all_train_train_line.append(all_train_neg_line[i])
        i=i+1

    while (i<cut2):
        all_train_test_line.append(all_train_neg_line[i])
        i=i+1
##############################################

######### partition positive first - otherwise oversample contamination ##
    num_pos=len(all_train_pos_line)
    pos_cut1=int(num_pos/2)
    pos_cut2=int(num_pos)

    tr_tr_pos=[]
    tr_te_pos=[]

    i=0
    while (i<pos_cut1):
        tr_tr_pos.append(all_train_pos_line[i])
        i=i+1

    while (i<pos_cut2):
        tr_te_pos.append(all_train_pos_line[i])
        i=i+1
##########################################################################    


########## oversample positive ####################################
######### balance positive-negative samples in train/train-test set #######
    i=0
    while (i<cut1*2):
        all_train_train_line.append(tr_tr_pos[int(i%len(tr_tr_pos))])
        i=i+1
#        if(int(i%len(tr_tr_pos))==0): # oversample positive
#            random.shuffle(tr_tr_pos)

    while (i<cut2*2):
        all_train_test_line.append(tr_te_pos[int(i%len(tr_te_pos))])
        i=i+1
 #       if(int(i%len(tr_te_pos))==0): # oversample positive
 #           random.shuffle(tr_te_pos)
###############################################################33

    random.shuffle(all_train_train_line)
    random.shuffle(all_train_test_line)

    print(len(all_train_train_line))
    print(len(all_train_test_line))

    return (all_train_train_line, all_train_test_line)

