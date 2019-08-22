#!/bin/bash


test_path=datasets/test-data/test-xml/UCCA_English-Wiki2
save_path=./exp/multilingual-lexical/emb_drop/emb_drop05
pred_path=$save_path/UCCA_English-Wiki2

alignments_path=embedding/alignment.txt

gpu=2
python -u run.py predict\
    --gpu=$gpu \
    --test_path=$test_path \
    --pred_path=$pred_path \
    --save_path=$save_path \
    --alignments=$alignments
