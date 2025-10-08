seed=${1:-55}
model_path=${2:-"../pretrained_models/llava-v1.5-7b"}
dataset_file=${3:-"../../CC3M/merged_cc3m.json"}
image_folder=${4:-"../../CC3M/cc3m_jpg"}
save_path=${5:-"./activations"}

CUDA_VISIBLE_DEVICES=0 python activation_collector.py \
--model-path ${model_path} \
--dataset-file ${dataset_file} \
--image-folder ${image_folder} \
--seed ${seed} \
--save_path ${save_path} \
