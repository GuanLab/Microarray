#!/bin/bash

for i in 0
do
    
    python3 train_2d_simple_generator-cv5.py |tee -a log_0_cv5.txt
    cp weights_cv5.h5 weights_0_cv5.h5

    python3 train_2d_simple_generator-cv5.py |tee -a log_1_cv5.txt
    cp weights_cv5.h5 weights_1_cv5.h5

    python3 train_2d_simple_generator-cv5.py |tee -a log_2_cv5.txt
    cp weights_cv5.h5 weights_2_cv5.h5


    python3 train_2d_simple_generator-cv5.py |tee -a log_3_cv5.txt
    cp weights_cv5.h5 weights_3_cv5.h5

    python3 train_2d_simple_generator-cv5.py |tee -a log_4_cv5.txt
    cp weights_cv5.h5 weights_4_cv5.h5
done
