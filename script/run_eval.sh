#!/bin/bash


# define fixed variables
dataset="imagenet100";
latent_class="hier";
meta_file_train="meta_data_train.csv";

ckpt=$1;
eval_epoch=$2;

command="python ../clinfonce/main_linear.py --ckpt "$ckpt/ckpt_epoch_$eval_epoch.pth" \
--learning_rate 0.3 \
--lr_scheduling cosine \
--epochs 100 --latent_class $latent_class \
--meta_file_train $meta_file_train \
--method SupCon --batch_size 128 \
--num_workers 10 \
"

echo $command;
eval $command;



