# this R code is for probe intensity recovery 
# author: Yanan Qin



#read the probe intensities of all cel files and calculate the mean matrix
library('affxparser')

count =0
file =  read.table("/media/ssd/yananq/mace/pos_recover_dir.txt", sep = "\n")
ref = matrix(0, 1164, 1164)
for ( i in seq(1, nrow(file), by=1)){
    celpath = file[i, 1]
    a = readCelRectangle(as.character(celpath), xrange=c(0, Inf), yrange=c(0, Inf), asMatrix=TRUE, readOutliers=FALSE)
    if (sum(is.nan(a$intensities)) == 0){ #in case of intensity is NaN
        ref = ref+a$intensities
        count = count +1
    }
}

count
ref = ref/count#count =35043

write.table(ref, file="/media/ssd/yananq/mace/reference.txt", row.names=FALSE, col.names=FALSE)


#read all contmainated cel files and save the intensities of each of them into a txt file 
#for python code use
# can read 1703 cel files into txt
file_contam =  read.table("/media/ssd/yananq/mace/contam_dir.txt", sep = "\n") 

for ( i in seq(1, nrow(file_contam), by=1)){
    #catch the error:Error in readCelHeader(filename) :
    #   [affxparser Fusion SDK exception] Failed to parse header of CEL file: 
    #         /media/ssd/yananq/mace/bg/E-GEOD-12400.raw.1/GSM308488.CEL
    try({ 
        celpath = file_contam[i, 1]
        txt_name = as.list(strsplit(as.character(celpath), split='/', fixed=TRUE))[[1]][8]
    
        a = readCelRectangle(as.character(celpath), xrange=c(0, Inf), yrange=c(0, Inf), asMatrix=TRUE, readOutliers=FALSE)
        write.table(a$intensities, file=paste("/media/ssd/yananq/mace/contam_intensities/", txt_name, ".txt", sep=''), row.names=FALSE, col.names=FALSE)
      }, silent=TRUE)
}

