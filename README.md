# Contamination in Published Microarray Studies Retrospectively Detected by Deep Learning
In the Microarray study, we retrospectively constructed 37,724 raw microarray images, and developed a deep learning algorithm to automatically detect defects in the original images. We provided the identified problematic arrays, affected genes and the imputed arrays as well as software tools to scan for such contamination as a resource to the research community to help future studies to scrutinize and critically analyze these data. 
Please contact (yananq@umich.edu or gyuanfan@umich.edu) if you have any questions or suggestions.

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

## If you want to repeat our experiments:
### Create positive labels 
Download folder 'make_label' and run the following command:
```
perl Pick.pl
perl Split_processed.pl
python3 label.py
```

### Train 
Open split_5cv.ipynb and run the code inside. This step is to split the data set into 5 folds for the futural five-fold Cross Validation.
For training, directly run 
```
bash bash-cv1.sh
bash bash-cv2.sh
bash bash-cv3.sh
bash bash-cv4.sh
bash bash-cv5.sh
```

### Prediction
Users can run prediction code using our saved model. Download the 'predict' folder, then
```
python3 pred-cv1.py
python3 pred-cv2.py
python3 pred-cv3.py
python3 pred-cv4.py
python3 pred-cv5.py
```
Then open dice_index.ipynb to calculate the dice coefficient

### Expression data in contaminated regions showed less coordinance within probesets
Download folder 'compare_var', and open gene_expression.ipynb, run the code inside. Then
```
R get_RMA.R
python3 Var_compare.py
```
Open and run Var_compare.ipynb, you can access the result.

### Literature mining
Download folder 'crawl' and run crawl.py using the following command 
```
python3 crawl.py
```


### Recover Contaminated Probes
Download folder 'recover' and run the code in recover.ipynb. Inside the ipynb file, there are some commands to ask you run R and python code:
```
R recover.R
python3 recover_intensity.py 
python3 recover_save_to_map_txt.py
```
Contam_list.ipynb is to give the list of contaminated genes in samples tested as positive



## If you want to test whether your sample has contamination
```
```
## If you want to use our recovered microarray data
