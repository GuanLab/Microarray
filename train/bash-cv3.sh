#!/bin/bash

for i in 0
do
    
    python3 train_2d_simple_generator-cv3.py |tee -a log_0_cv3.txt
    cp weights_cv3.h5 weights_0_cv3.h5

    python3 train_2d_simple_generator-cv3.py |tee -a log_1_cv3.txt
    cp weights_cv3.h5 weights_1_cv3.h5

    python3 train_2d_simple_generator-cv3.py |tee -a log_2_cv3.txt
    cp weights_cv3.h5 weights_2_cv3.h5


    python3 train_2d_simple_generator-cv3.py |tee -a log_3_cv3.txt
    cp weights_cv3.h5 weights_3_cv3.h5

    python3 train_2d_simple_generator-cv3.py |tee -a log_4_cv3.txt
    cp weights_cv3.h5 weights_4_cv3.h5

    python3 pred-cv3.py
    python3 pred_random-cv3.py
done
