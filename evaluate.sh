#!/bin/bash

gold_path=datasets/test-data/test-xml-gold/UCCA_English-Wiki
save_path=./exp/multilingual-lexical/english/parser2

gpu=2
python -u run.py evaluate\
    --gpu=$gpu \
    --gold_path=$gold_path \
    --save_path=$save_path
