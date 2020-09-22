# Contamination in Published Microarray Studies Retrospectively Detected by Deep Learning

## Installation
```
git clone https://github.com/GuanLab/Microarray.git
```

## Required dependencies
* [python](https://www.python.org) (3.7.7)
* [tensorflow](https://www.tensorflow.org/) (1.14.0) A popular deep learning package. It can be installed by:
```
conda install tensorflow-gpu=1.14
```
* [keras](https://keras.io/) (2.2.4) A popular deep learning package using tensorflow backend. It can be installed by:
```
conda install keras-gpu=2.2.4
```
* [opencv-python](https://pypi.org/project/opencv-python/)
```
pip install opencv-python 
```
* [numpy](http://www.numpy.org/)
* [pandas](https://pypi.org/project/pandas/)
* [json](https://docs.python.org/3/library/json.html)
* [os](https://docs.python.org/3/library/os.html)
* [bs4](https://pypi.org/project/bs4/)
* [requests](https://pypi.org/project/requests/2.7.0/)
* [re](https://docs.python.org/3/library/re.html)
* R (3.6.1)
* [Jupyter Notebook](https://jupyter.org/)
## Dataset
The data can be directly downloaded from our web server:
* Microarray positive labels: contains 842 human-labelled positive samples
* Microarray raw images: contains 37724 raw microarray images
* Microarray positive prediction results: contains 1810 positive prediction results 
* Microarray recovered probe expression: contains 1703 recovered samples 
* Microarray CEL files: raw CEL files


## Train 
Open split_5cv.ipynb and run the code inside. This step is to split the data set into 5 folds 

## Prediction
Users can run prediction code using out saved model.


## Recover Contaminated Probes
