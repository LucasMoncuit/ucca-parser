#!/bin/bash

train_path=datasets/train-dev-data/train-xml/UCCA_English-Wiki
dev_path=datasets/train-dev-data/dev-xml/UCCA_English-Wiki

en_train_path=datasets/trainSet/UCCA_English-Wiki
en_dev_path=train-dev-data/devSet/UCCA_English-Wiki

de_train_path=datasets/train-dev-data/train-xml/UCCA_German-20K
de_dev_path=datasets/train-dev-data/dev-xml/UCCA_German-20K

fr_train_path=datasets/train-dev-data/train-xml/UCCA_French-20K
fr_dev_path=datasets/train-dev-data/dev-xml/UCCA_French-20K

emb_path=embedding/cc.en.300.vec
save_path=./exp/multilingual-lexical/finaltest_en
config_path=./config.json

en_test_wiki_path=test-data/test-xml-gold/UCCA_English-Wiki
en_test_20k_path=test-data/test-xml-gold/UCCA_English-20K
fr_test_20k_path=test-data/test-xml-gold/UCCA_French-20K
de_test_20k_path=test-data/test-xml-gold/UCCA_German-20K

Alignment=embedding/alignment.txt

Parallel=True



gpu=1

if [ ! -d "$save_path" ]; then
    mkdir -p "$save_path"
    cp $config_path $save_path
fi
export PYTHONPATH=$PWD
python -u run.py train\
    --gpu=$gpu \
    --save_path=$save_path \
    --train_path=$train_path \
    --test_wiki_path=$test_wiki_path \
    --test_20k_path=$test_20k_path \
    --dev_path=$dev_path \
    --emb_path=$emb_path \
    --config_path=$config_path \
    --en_train_path=$en_train_path \
    --en_dev_path=$en_dev_path \
    --de_train_path=$de_train_path \
    --de_dev_path=$de_dev_path \
    --fr_train_path=$fr_train_path \
    --fr_dev_path=$fr_dev_path \
    --parallel=$Parallel \
    --alignment=$Alignment