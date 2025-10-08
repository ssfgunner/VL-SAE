import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from hooks import InputHook, OutputHook

from PIL import Image
import math

from transformers import set_seed

def activation_allocation(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    with open(args.dataset_file, "r") as f:
        datasets = json.load(f)

    feature_dict = {}
    image_features_list = []
    text_features_list = []
    image_files = []
    texts = []
    for idx, line in tqdm(enumerate(datasets), desc="Inference..."):
        image_file = line["key"] + '.jpg'
        text = line["caption"]
        if model.config.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        target_layer_name = args.target_layer_name
        image_tokens_seq_len = 576 # number of image tokens for 224x224 images with 16x16 patch size

        with InputHook(model, outputs=[target_layer_name], as_tensor=True) as h:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=1,
                    use_cache=True)
            returned_features = h.layer_outputs[target_layer_name][0]
            image_start_idx = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].item()
            image_features = returned_features[:, image_start_idx:image_start_idx+image_tokens_seq_len, :].mean(0).mean(0)
            text_features = returned_features[:, image_start_idx+image_tokens_seq_len:, :].mean(0).mean(0)
        image_features_list.append(image_features.detach().cpu().numpy())
        text_features_list.append(text_features.detach().cpu().numpy())
        image_files.append(image_file)
        texts.append(text)
        # break
        
    feature_dict = {
        'image_features': image_features_list,
        'text_features': text_features_list,
        'image_file': image_files,
        'text': texts,
    }

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    save_path = os.path.join(args.save_path, f"llava_cc3m_{target_layer_name}_embeddings.pt")
    torch.save(feature_dict, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--dataset-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--target_layer_name", type=str, default="model.layers.30")
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument('--save_path', default="./", type=str)

    args = parser.parse_args()
    set_seed(args.seed)
    activation_allocation(args)
