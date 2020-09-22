# author: Yanan Qin 
'''
This py code is to crawl articles from GEO website
'''
import numpy as np
import pandas as pd
import seaborn as sns
import json
import shutil
import random
from os import path
import nibabel as nib
import glob
import os
import sys
# import component_cut
import cv2
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from tensorflow import Graph
from tensorflow import Session
from bs4 import BeautifulSoup
import requests
import re
from keras.backend.tensorflow_backend import set_session


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

count_image = 0
count_set = 0
pos_set = set()  #set of positive samples


for i in os.listdir('/media/ssd2/yananq/mace/code/pos_pred/'):
    count_image +=1
    pos_set.add('.'.join(i.split('.')[:-2]))
    
print(count_image/37724, ' of the images are affected by such image-level defects')  #0.04798006574064256



count_set = set() # count how many datasets are using the 37724 samples
pos_dataset = set() #count how many datasets have contaminations
full_ls = os.listdir('/media/ssd2/yananq/mace/data/raw_image/mace_full_png') #full list of 37724 images

for i in os.listdir('/media/ssd/yananq/mace/bg'):
    for j in os.listdir('/media/ssd/yananq/mace/bg/'+i):
        if j+'.png' in full_ls:
            count_set.add(i)
            if j in pos_set:
                pos_dataset.add(i)
print(len(count_set), 'how many datasets are using the 37724 samples')   #3165             
print(len(pos_dataset), 'how many datasets have contaminations') # 846


json = pd.read_json("/media/ssd2/yananq/mace/code/baseline/probe_correspondant.json")
map_dict  = json[0:54675]
probe2gene = map_dict.to_dict()['probe2gene']
full_genes = set(map_dict['probe2gene'].dropna())


#building a matrix with probe set ids's at their corresponding location
probe_mat = np.empty((1164, 1164), dtype=np.dtype('U100'))
for index, row in map_dict.iterrows():
    probe_set = row.name
    x_loc = row['dict'][0]
    y_loc = row['dict'][1]
    for i in range(len(x_loc)):
        probe_mat[x_loc[i],y_loc[i]] = probe_set


paper2genes = dict()  # key: name of study/dataset; value: set of all genes mentioned in the paper
paper2contamgenes = dict()   # key: name of study/dataset; value: set of all contaminated genes mentioned in the paper


def f(element, thresh):
    '''
    input: pixel value; threshold 
    output: 1 if pixel value is greater than threshold, 0 otherwise 
    '''
    return 1 if element >= thresh else 0
f = np.vectorize(f) 


################################### for i in all contaminated data sets:
for i in pos_dataset:
    # map data set name to series name 
    series = 'GSE'+ i.split('.')[0][-5:]    
    # open website 
    web ='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=' +series
    # crawl down paper (by PMCID)  # note: the citation might be missing, so we do try and catch!
    try: 
        # go to series web page
        page = requests.get(web)
        soup = BeautifulSoup(page.content, 'html.parser')
        pmid = soup.find('span',{'class':'pubmed_id'}).text

        # go to pmid webpage
        publication_web = 'https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/'
        publication_page = requests.get(publication_web)
        publication_soup = BeautifulSoup(publication_page.content, 'html.parser')
        link = publication_soup.find('a', {'class':'id-link'})['href']

        # get full text
        text = requests.get(link, headers={'User-Agent': 'Mozilla/5.0'})
        text = requests.get(text.url, headers={'User-Agent': 'Mozilla/5.0'}) #make sure to get the final url if redirect
        text_soup = BeautifulSoup(text.content, 'html.parser')
        full_text = text_soup.text

        gene_contam = set() #name set of all contaminated genes in the study i
        if paper2genes.get(i) is None:
            paper2genes[i] = set()
        if paper2contamgenes.get(i) is None:
            paper2contamgenes[i] = set()

        # mine genes mentioned in the paper
        for gene in full_genes:
            if gene in full_text:
                #  save into paper2genes
                paper2genes.get(i).add(gene)

        if len(paper2genes.get(i))!=0:  # if the paper mentioned any genes
            for j in os.listdir('/media/ssd/yananq/mace/bg/'+i): #iterate over samples in study i
                if j+'.png.npy' in os.listdir('/media/ssd2/yananq/mace/code/pos_pred'): # if sample j is contaminated
                    pred_new = np.load('/media/ssd2/yananq/mace/code/pos_pred/'+j+'.png.npy')
                    pred_new = f(pred_new, 0.5)
                    # get the name set probes with contamination
                    probe_contam = set(probe_mat[pred_new==1])
                    probe_contam = [i for i in list(probe_contam) if i != '']   #remove ''

                    #get the name set of contmainated gene in this sample and add to gene_contam
                    gene_contam |= set(probe2gene[k] for k in probe_contam if str(probe2gene[k])!= 'nan')

            # calculate how many of them where in contaminated areas
            for gene in gene_contam: #iterate over the contaminated genes
                if gene in paper2genes.get(i): # if the gene was mentioned in the paper
                    # save into paper2contamgenes
                    paper2contamgenes.get(i).add(gene)
    except:
        pass
############################end of for loop

# count how many contaminated genes mentioned in total (count duplicates)
total_contamgenes_mentioned = np.sum(len(val) for val in paper2contamgenes.values())
# count how manygenes mentioned in total (count duplicates)
total_genes_mentioned = np.sum(len(val) for val in paper2genes.values())
# calculate XX% of them were in contaminated areas (count duplicates)
print(total_contamgenes_mentioned/total_genes_mentioned, 'of genes were in contaminated areas') # 0.28816168763102723
print(total_contamgenes_mentioned) # 8797
print(total_genes_mentioned) #30528
print(len(paper2contamgenes.keys())) # 480
print(len(paper2genes.keys())) # 480
