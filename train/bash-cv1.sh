#!/bin/bash

for i in 0
do
    
    python3 train_2d_simple_generator-cv1.py |tee -a log_0_cv1.txt
    cp weights.h5 weights_0.h5

    python3 train_2d_simple_generator-cv1.py |tee -a log_1_cv1.txt
    cp weights.h5 weights_1.h5

    python3 train_2d_simple_generator-cv1.py |tee -a log_2_cv1.txt
    cp weights.h5 weights_2.h5


    python3 train_2d_simple_generator-cv1.py |tee -a log_3_cv1.txt
    cp weights.h5 weights_3.h5

    python3 train_2d_simple_generator-cv1.py |tee -a log_4_cv1.txt
    cp weights.h5 weights_4.h5
done
