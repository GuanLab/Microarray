# author: Yanan Qin
library(BiocInstaller)
biocLite(c("affy", "simpleaffy"))
install.packages(c("tcltk", "scales"))
library(affy)
library(tcltk)
library(oligo)


file =  read.table("/media/ssd/yananq/mace/sample_dir.txt", header=F)

 for ( i in seq(1, nrow(file), by=1)){
	celpath = file[i, 1]
	list = list.files(as.character(celpath),full.names=TRUE)
	data = read.celfiles(list)
	data.rma = rma(data)
	sampleNames(data.rma) <- gsub(".CEL$", "", sampleNames(data.rma))
	data.matrix = exprs(data.rma)
	write.exprs(data.rma,file=paste("/media/ssd/yananq/mace/cel_expression/", substr(celpath, 27, nchar(as.character(celpath))), ".txt", sep=''))
}

# finally doing rma on 1455 dirs
# 1533-1455 = 78 dirs have more than 1 platforms
