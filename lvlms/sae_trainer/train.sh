#!/usr/bin/env bash 
pretrained_model=${1:-"../pretrained_models/llava-v1.5-7b"}
topk=${2:-"256"}
hidden_ratio=${3:-"8"}
save_path=${4:-"./sae_weights"}
aux_ae_path=${5:-"./sae_weights/llava_aux_best.pt"}
embeddings_path=${6:-"../representation_collection/activations/llava_cc3m_activations_model.layers.30_mean.pt"}

train auxiliary autoencoder
python train_aux.py --embeddings_path ${embeddings_path} --save_path ${save_path} 

# train VL-SAE
python train.py --pretrained_model ${pretrained_model} --aux_ae_path ${aux_ae_path} \
    --topk ${topk} --hidden_ratio ${hidden_ratio} --save_path ${save_path} \
    --embeddings_path ${embeddings_path} --initial_lr 1e-4 \