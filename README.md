# Contamination in Published Microarray Studies Retrospectively Detected by Deep Learning
In the Microarray study, we retrospectively constructed 37,724 raw microarray images, and developed a deep learning algorithm to automatically detect defects in the original images. Here we provide the identified problematic arrays, affected genes and the imputed arrays as well as software tools to scan for such contamination as a resource to the research community to help future studies to scrutinize and critically analyze these data. 
Please contact (yananq@umich.edu or gyuanfan@umich.edu) if you have any questions or suggestions.
![Figure1](Figure/Fig2.png?raw=true "Title")


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
* [Perl](https://www.perl.org/) (v5.26.1)


## Dataset
The data can be directly downloaded from [https://osf.io/g4qxu/?view_only=3aaf0f0469744e54befbc4f86143ab47] on Open Science Framework:
* recovered v1-v6: contains 1703 recovered samples 
* Contam_genes: contains the list of contaminated genes for every positive sample
* label: human-labelled microarray images
* model: our deep learning model
* HG-U133_Plus_2.cdf: cdf file for HG-U133_Plus_2 platform

## Codes for experiments 

### Recovering of microarray images from CEL files
Download the folder 'reconstruct_to_image' and run the following command:
```
python3 setup.py build
python3 test_full.py
```

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
Rscript get_RMA.R
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
Rscript recover.R
python3 recover_intensity.py 
python3 recover_save_to_map_txt.py
```
Contam_list.ipynb is to give the list of contaminated genes in samples tested as positive



## Vignettes -- if you want to test whether your sample has contamination
Have the microarray CEL file in hand and download Github folder 'Example', our deep learning model and HG-U133_Plus_2.cdf from OSF. Move the model and cdf file into 'Example'.<br />
Before doing the first step, make sure you have changed the directory in cel_to_img.py. 
First of all, you would need to recover your microarray cel file to an image:
```
python3 setup.py build 
python3 cel_to_img.py
```
The result is a image png file converted from microarray CEL file. 

In the prediction step, we need to use GPU in order to achieve Then run the following command:
```
CUDA_VISIBLE_DEVICES=XX python3 Example/pred_example.py
```
This command will have two resulting files:
* result.npy: the prediction result generated by our model
* contam_genes.txt: contains the genes whose probes locats at contaminated area
In order to recover the contaminated region, you should have the original CEL file in hand, and run the following command:
```
Rscript Example/recover_example.R
python3 Example/recover_example.py
```
Then recovered.txt will be the recovered probe intensities.

## If you want to use our recovered microarray data
Our recovered data is available to be downloaded from Open Science Framework. Note that our recovered data only has 1703 microarray samples with defects, so please check data availability before using.
* Microarray recovered probe expression: contains 1703 recovered samples. The recovered probe expressions of every sample were saved in a TXT file with the same name as CEL file. We save the recoverd value as: probe id: value. 
* contam_genes: contains the list of contaminated genes for every positive sample
For example, if your CEL is called XX.CEL, XX.CEL.txt in Microarray recovered probe expression contains recovered probe expression and XX.CEL.txt in contam_genes contains a list of genes with defects.

## FAQ
* My microarray sample does not appear in the set of 1,810 contaminated samples, does that mean my sample does not have contamination?<br />
Answer: No. We only tested 37,724 microarray samples published before 2019. To get a better result, we encourage you follow our instruction and use our model to test your microarray sample.
* Can the deep learning model be applied to all platforms of microarrays? <br />
Answer: No. We used microarray images of Hg-U133 Plus2 platform to train our model. The model validity on other platforms are not available yet.
* The code does not work. What should I do?<br />
Answer: We recommend that you check all the packages installed and check whether you have changed all the directories in the code. If that does not solve your problem, please create a ticket on the GitHub Issue Tracker.


