#!/bin/bash


test_path=datasets/test-data/test-xml/UCCA_English-Wiki
save_path=./exp/multilingual-lexical/english
pred_path=$save_path/UCCA_English-Wiki

gpu=2
python -u run.py predict\
    --gpu=$gpu \
    --test_path=$test_path \
    --pred_path=$pred_path \
    --save_path=$save_path
