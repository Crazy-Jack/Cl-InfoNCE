#!/bin/bash


# define fixed variables
dataset="imagenet100";
instruction="hier";
meta_file_train="meta_data_train.csv";

ckpt=$1;

command="python ../clinfonce/main_linear.py --ckpt $ckpt \
--learning_rate 0.3 \
--lr_scheduling cosine \
--epochs 100 --instruction $instruction \
--meta_file_train $meta_file_train \
--method SupCon --batch_size 128 \
--num_workers 10 \
"

echo $command;
eval $command;



