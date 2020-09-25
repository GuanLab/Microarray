# this R code is for probe intensity recovery 
# author: Yanan Qin
library('affxparser')
celpath = './GSM707032.CEL'  # change the path of your CEL file
a = readCelRectangle(as.character(celpath), xrange=c(0, Inf), yrange=c(0, Inf), asMatrix=TRUE, readOutliers=FALSE)
write.table(a$intensities, file='to_recover.txt', row.names=FALSE, col.names=FALSE)


