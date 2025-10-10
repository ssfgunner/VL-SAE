seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"adversarial"}
model_path=${4:-"../../pre-trained_models/llava-v1.5-7b"}
sae_path=${5:-"../../sae_trainer/sae_weights/llava_256_8_best.pth"}
cd_alpha=${6:-0.6}
cd_beta=${7:-0.8}
modify_alpha=${8:-0.8}
modify_beta=${9:-0.9}
alignment_path=${10:-"../../sae_trainer/sae_weights/llava_aux_best.pt"}
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=./data/coco/val2014
else
  image_folder=./data/gqa/images
fi

python ./eval/inference_vlsae_llava_pope.py \
--model-path ${model_path} \
--question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--seed ${seed} \
--answers-file ./output/llava15_${dataset_name}_pope_${type}_answers_vlsae_seed${seed}_${modify_alpha}.jsonl \
--extract_layers "model.layers.30" \
--top_k_sae 256 \
--cd_alpha ${cd_alpha} \
--cd_beta ${cd_beta} \
--hidden_ratio_sae 8 \
--sae-path ${sae_path} \
--alignment-path ${alignment_path} \
--modify_alpha ${modify_alpha} \
--modify_beta ${modify_beta} \
--use_sae \

python ./eval/eval_pope.py --gt_files ./data/POPE/coco/coco_pope_${type}.json \
    --gen_files ./output/llava15_${dataset_name}_pope_${type}_answers_vlsae_seed${seed}_${modify_alpha}.jsonl

