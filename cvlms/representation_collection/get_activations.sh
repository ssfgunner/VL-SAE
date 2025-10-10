#!/usr/bin/env bash 
pretrained_model=${1:-"ViT-B-32"}
model_path=${2:-"laion2b_s34b_b79k"}
# for local checkpoint
# ckpt_path=${2:-"../../pretrained_models/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin"}
text_path=${3:-"../../CC3M/merged_cc3m.json"}
image_path=${4:-"../../CC3M/cc3m_jpg"}
save_path=${5:-"./activations"}

python activation_collector.py --pretrained_model ${pretrained_model} --model_path ${model_path} \
   --text_path ${text_path} --save_path ${save_path}

# python activation_collector.py --pretrained_model ${pretrained_model} --model_path ${model_path} \
#    --image_path ${image_path} --save_path ${save_path}