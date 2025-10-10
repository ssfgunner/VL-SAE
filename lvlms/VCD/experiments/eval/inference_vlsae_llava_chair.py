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

from sae_model import VL_SAE, AuxiliaryAE
import random

from PIL import Image
import math
from functools import partial

# import kornia
from transformers import set_seed
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
from vcd_utils.vlsae_sample import evolve_vlsae_sampling

evolve_vlsae_sampling()

class FeatureExtractor:
    def __init__(self, model, layer_names, sae_path=None, alignment_path=None, input_dim=4096, topk=128, hidden_ratio=8, modify_fn=None):
        self.model = model
        self.features = {}
        self.layer_names = layer_names if isinstance(layer_names, (list, tuple)) else [layer_names]
        self.hooks = {}
        self.modify_fn = modify_fn
        self.input_dim = input_dim
        self.vis_indices = None
        
        if sae_path:
            self.sae = VL_SAE(input_dim, hidden_dim=hidden_ratio*input_dim, topk=topk).cuda().half()
            self.alignment_model = AuxiliaryAE(input_dim, input_dim, projection_dim=4096).cuda().half()
            self.sae.load_state_dict(torch.load(sae_path))
            print(f'Successfully loaded SAE model from {sae_path}')
            self.alignment_model.load_state_dict(torch.load(alignment_path))
            print(f'Successfully loaded Auxiliary AE model from {alignment_path}')
            self.sae.eval()
            self.alignment_model.eval()

    def set_model(self, model):
        self.model = model

    def hook_fn(self, name):
        def hook(module, input, output):
            if self.vis_indices is not None:
                hidden_states = output[0]
                
                vis_features = hidden_states[:, self.vis_indices, :]
                text_features = hidden_states[:, self.vis_indices[-1]+1:, :]
                vis_features_mean = vis_features.mean(dim=1) 
                text_features_mean = text_features.mean(dim=1)
                
                if self.modify_fn is not None:
                    modified_vis_features_mean, modified_text_features_mean = self.modify_fn(
                        vis_features_mean, text_features_mean, self.sae, self.alignment_model
                    )
                    hidden_states[:, self.vis_indices, :] = hidden_states[:, self.vis_indices, :] + \
                        modified_vis_features_mean.unsqueeze(1) - vis_features_mean.unsqueeze(1)
                    hidden_states[:, self.vis_indices[-1]+1:, :] = hidden_states[:, self.vis_indices[-1]+1:, :] + \
                        modified_text_features_mean.unsqueeze(1) - text_features_mean.unsqueeze(1)
                    return (hidden_states,)
            return output
        return hook
    
    def _get_layer(self, name):
        if '.' in name:
            module = self.model
            for part in name.split('.')[:-1]:
                module = getattr(module, part)
            return getattr(module, name.split('.')[-1])
        return getattr(self.model, name)

    def add_hooks(self, layer_names=None):
        if layer_names is None:
            layer_names = self.layer_names
        elif isinstance(layer_names, str):
            layer_names = [layer_names]
            
        for name in layer_names:
            if name not in self.hooks:
                layer = self._get_layer(name)
                self.hooks[name] = layer.register_forward_hook(self.hook_fn(name))
                # print(f"Added hook to layer: {name}")
    
    def remove_hooks(self, layer_names=None):
        if layer_names is None:
            layer_names = list(self.hooks.keys())
        elif isinstance(layer_names, str):
            layer_names = [layer_names]
            
        for name in layer_names:
            if name in self.hooks:
                self.hooks[name].remove()
                del self.hooks[name]
                # print(f"Removed hook from layer: {name}")
    
    def set_vis_indices(self, indices):
        self.vis_indices = indices

def directly_sae_forward(vision_features, text_features, sae):
    vision_features, text_features = sae(vision_features, text_features)
    return vision_features, text_features

def weight_sae_forward(vision_features, text_features, sae, alpha=0.1):
    recon_vision_features, recon_text_features = sae(vision_features, text_features)
    return vision_features + alpha * recon_vision_features, text_features + alpha * recon_text_features

def image_constrain_text(vision_features, text_features, sae, alignment_model, alpha=1, beta=0.5):
    vision_embeds, text_embeds = alignment_model.encoder(vision_features, text_features)
    vision_embeds, text_embeds = sae.encode(vision_embeds), sae.encode(text_embeds)
    text_embeds = (1-beta)*text_embeds+beta*vision_embeds
    recon_text_features = alignment_model.decoder(text_embed=sae.text_decoder(text_embeds))[-1]
    text_features = (1-alpha)*text_features + alpha*recon_text_features
    return vision_features, text_features

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    modify_fn = partial(image_constrain_text, alpha=args.modify_alpha, beta=args.modify_beta)

    feature_extractor = FeatureExtractor(None, args.extract_layers, sae_path=args.sae_path, alignment_path=args.alignment_path, topk=args.top_k_sae, hidden_ratio=args.hidden_ratio_sae, modify_fn=modify_fn)

    img_files = os.listdir(args.data_path)
    random.shuffle(img_files)

    with open(args.data_path + '../annotations/instances_val2014.json', 'r') as f:
        lines = f.readlines()
    coco_anns = json.loads(lines[0])

    img_dict = {}

    categories = coco_anns["categories"]
    category_names = [c["name"] for c in categories]
    category_dict = {int(c["id"]): c["name"] for c in categories}

    for img_info in coco_anns["images"]:
        img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

    for ann_info in coco_anns["annotations"]:
        img_dict[ann_info["image_id"]]["anns"].append(
            category_dict[ann_info["category_id"]]
        )


    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for img_id in tqdm(list(range(len(img_files)))[:500]):
        img_file = img_files[img_id]
        img_id = int(img_file.split(".jpg")[0][-6:])
        img_info = img_dict[img_id]
        assert img_info["name"] == img_file
        img_anns = set(img_info["anns"])
        img_save = {}
        img_save["image_id"] = img_id

        qs = "Please describe this image in detail."
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
        image_path = args.data_path + img_file

        image = Image.open(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # print('input_ids:', input_ids.shape)
        
        image_start_idx = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].item()
        image_tokens_seq_len = 576
        vis_token_indices = list(range(image_start_idx, image_start_idx+image_tokens_seq_len))
            
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
                do_sample=True,
                temperature=args.temperature,
                use_sae=args.use_sae,
                feature_extractor=feature_extractor,
                vis_indices=vis_token_indices,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=512,
                use_cache=False)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        img_save["caption"] = outputs
        ans_file.write(json.dumps(img_save) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="/data/shufan/shufan/VLM_SAE/CLIP_benchmark/root/val2014/", help="data path")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--sae-path", type=str, default="")
    parser.add_argument("--alignment-path", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_k_sae", type=int, default=128)
    parser.add_argument("--hidden_ratio_sae", type=int, default=8)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_sae", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extract_layers", type=str, default="",
                        help="features of extracted layers, e.g., encoder.layer.11,decoder.layer.5")
    parser.add_argument("--sae_path", type=str, default="")
    parser.add_argument("--modify_alpha", type=float, default=0.1,
                        help="alpha for modifying features")
    parser.add_argument("--modify_beta", type=float, default=4,)

    # parser.add_argument("--modification_type", type=str, default="noise",
    #                     help="type of modification, e.g., noise, scaling")
    args = parser.parse_args()
    
    
    set_seed(args.seed)
    eval_model(args)
