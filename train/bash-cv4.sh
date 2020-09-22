#!/bin/bash

for i in 0
do
    
    python3 train_2d_simple_generator-cv4.py |tee -a log_0_cv4.txt
    cp weights_cv4.h5 weights_0_cv4.h5

    python3 train_2d_simple_generator-cv4.py |tee -a log_1_cv4.txt
    cp weights_cv4.h5 weights_1_cv4.h5

    python3 train_2d_simple_generator-cv4.py |tee -a log_2_cv4.txt
    cp weights_cv4.h5 weights_2_cv4.h5


    python3 train_2d_simple_generator-cv4.py |tee -a log_3_cv4.txt
    cp weights_cv4.h5 weights_3_cv4.h5

    python3 train_2d_simple_generator-cv4.py |tee -a log_4_cv4.txt
    cp weights_cv4.h5 weights_4_cv4.h5

    python3 pred-cv4.py
    python3 pred_random-cv4.py
done
