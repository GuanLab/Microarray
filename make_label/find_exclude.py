import cv2
import numpy as np
import os
import glob



all_label=glob.glob('split_processed/*')

FILE=open('exclude.txt','w')
for the_label in all_label:
    the_name=the_label.split('/')
#    print(('raw_image/'+the_name[-1]))
    img=cv2.imread('raw_image/mace_full/'+the_name[-1])
    (bbb,aaa)=(img.shape[0],img.shape[1])
    b=np.zeros((bbb,aaa))
    BAT=open(the_label,'r')
    for line in BAT:
        line=line.rstrip()
        table=line.split("\t");
## suppose image is 512X512, will be resized anyway for input
        table[1]=float(table[1])
        table[2]=float(table[2])

        polygon=[]
        y=int(round(float(table[1])*bbb))
        x=int(round(float(table[2])*aaa))
        #print(the_label,table[1],table[2],y,x)
        polygon.append([y,x])

        x=int(round(float(table[2])*aaa))

        y=int(round(float(table[1])*bbb+float(table[3])*bbb))
        #print(table[1],table[2],y,x)
        polygon.append([y,x])

        x=int(round(float(table[2])*aaa+float(table[4])*aaa))
        y=int(round(float(table[1])*bbb+float(table[3])*bbb))
        #print(table[1],table[2],y,x)
        polygon.append([y,x])
    
        x=int(round(float(table[2])*aaa+float(table[4])*aaa))
        y=int(round(float(table[1])*bbb))
        #print(table[1],table[2],y,x)
        polygon.append([y,x])
        poly = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(b, [poly], 1)



    size=aaa*bbb
    ratio=np.sum(b)/float(size)
    FILE.write(('%s\t%.4f\n') % (the_name[-1],ratio))
    print(the_name[-1],ratio)



