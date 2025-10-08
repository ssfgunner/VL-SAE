#!/usr/bin/env bash 
pretrained_model=${1:-"ViT-B-32"}
topk=${2:-"256"}
hidden_ratio=${3:-"8"}
save_path=${4:-"./sae_weights"}
text_embeddings_path=${5:-"../representation_collection/activations/ViT-B-32_text_embeddings.pt"}
image_embeddings_path=${6:-"../representation_collection/activations/ViT-B-32_vision_embeddings.pt"}

python train.py --pretrained_model ${pretrained_model} \
    --topk ${topk} --hidden_ratio ${hidden_ratio} --save_path ${save_path} \
    --text_embeddings_path ${text_embeddings_path} --image_embeddings_path ${image_embeddings_path} \
    --initial_lr 1e-3