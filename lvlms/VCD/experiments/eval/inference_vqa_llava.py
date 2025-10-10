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
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from sae_model import VL_SAE_COS
from PIL import Image
import math

# import kornia
from transformers import set_seed
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
from vcd_utils.vlsae_sample import evolve_vlsae_sampling
# evolve_vcd_sampling()

class FeatureExtractor:
    def __init__(self, model, layer_names, sae_path=None, input_dim=4096, topk=128, hidden_ratio=8, modify_fn=None):
        self.model = model
        self.features = {}
        self.layer_names = layer_names
        self.hooks = []
        self.modify_fn = modify_fn
        self.input_dim = input_dim
        self.vis_indices = None  # 存储视觉token的索引
        if sae_path:
            self.sae = VL_SAE_COS(input_dim, hidden_dim=hidden_ratio*input_dim, topk=topk).cuda().half()
            self.sae.load_state_dict(torch.load(sae_path))
            print(f'Successfully loaded SAE model from {sae_path}')
            self.sae.eval()
        def hook_fn(name):
            def hook(module, input, output):
                if self.vis_indices is not None:
                    # 分离视觉和文本特征
                    hidden_states = output[0]  # [batch_size, seq_len, hidden_dim]
                    
                    # 提取视觉和文本特征
                    vis_features = hidden_states[:, self.vis_indices, :]
                    text_features = hidden_states[:, self.vis_indices[-1]+1:, :]
                    vis_features_mean = vis_features.mean(dim=1)
                    text_features_mean = text_features.mean(dim=1)
                    # 如果需要修改特征
                    if self.modify_fn is not None:
                        modified_vis_features_mean, modified_text_features_mean = self.modify_fn(vis_features_mean, text_features_mean, self.sae)
                        hidden_states[:, self.vis_indices, :] = hidden_states[:, self.vis_indices, :] + modified_vis_features_mean.unsqueeze(1) - vis_features_mean.unsqueeze(1)
                        hidden_states[:, self.vis_indices[-1]+1:, :] = hidden_states[:, self.vis_indices[-1]+1:, :] + modified_text_features_mean.unsqueeze(1) - text_features_mean.unsqueeze(1)
                        return (hidden_states, )
                return output
            return hook
        
        for name in layer_names:
            # 根据层名获取对应层
            if '.' in name:
                module = self.model
                name_parts = name.split('.')
                for part in name_parts[:-1]:
                    module = getattr(module, part)
                layer = getattr(module, name_parts[-1])
            else:
                layer = getattr(self.model, name)
            
            # 注册钩子
            self.hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    def set_vis_indices(self, indices):
        """设置视觉token的索引"""
        self.vis_indices = indices

    def clear_features(self):
        self.features = {}
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def directly_sae_forward(vision_features, text_features, sae):
    # print('vision_features:', vision_features.shape)
    # print('text_features:', text_features.shape)
    vision_features, text_features = sae(vision_features, text_features)
    return vision_features, text_features

def weight_sae_forward(vision_features, text_features, sae, alpha=0.1):
    recon_vision_features, recon_text_features = sae(vision_features, text_features)
    return vision_features + alpha * recon_vision_features, text_features + alpha * recon_text_features

def image_constrain_text(vision_features, text_features, sae, alpha=0.01, beta=0.1):
    vision_sparse_features, text_sparse_features = sae.encode(vision_features), sae.encode(text_features)
    text_sparse_features = text_sparse_features + beta * vision_sparse_features
    recon_text_features = sae.text_decoder(text_sparse_features)
    text_features = text_features + alpha * recon_text_features
    return vision_features, text_features

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    if args.extract_layers:
        if args.modify_features:
            modify_fn = directly_sae_forward
            # modify_fn = weight_sae_forward
            # modify_fn = image_constrain_text
        else:
            modify_fn = None
        feature_extractor = FeatureExtractor(model, args.extract_layers.split(','), sae_path=args.sae_path, modify_fn=modify_fn, topk=args.top_k_sae, hidden_ratio=args.hidden_ratio_sae)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # print('input_ids:', input_ids.shape)
        
        # 获取视觉token的索引
        image_start_idx = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].item()
        image_tokens_seq_len = 576
        vis_token_indices = list(range(image_start_idx, image_start_idx+image_tokens_seq_len))
            
        
        # 设置视觉token索引
        if args.extract_layers:
            feature_extractor.set_vis_indices(vis_token_indices)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                images_cd=None,
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
                do_sample=True,
                temperature=args.temperature,
                use_sae=True,
                vis_indices=vis_token_indices,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=False)
            # print('output_ids:', output_ids.shape)
            # 保存提取的特征
            # if args.extract_layers:
                # features = feature_extractor.features
                # feature_path = os.path.join(args.feature_dir, f"{idx}_features.pt")
                # 保存特征时包含视觉token的索引信息
                # torch.save({
                #     'features': features,
                #     'vis_indices': vis_token_indices,
                # }, feature_path)
                # feature_extractor.clear_features()

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

    if args.extract_layers:
        feature_extractor.remove_hooks()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--sae-path", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_k_sae", type=int, default=128)
    parser.add_argument("--hidden_ratio_sae", type=int, default=8)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extract_layers", type=str, default="",
                        help="features of extracted layers, e.g., encoder.layer.11,decoder.layer.5")
    parser.add_argument("--feature_dir", type=str, default="features",
                        help="directory to save extracted features")
    parser.add_argument("--modify_features", action='store_true', default=False,
                        help="whether to modify extracted features")
    # parser.add_argument("--modification_type", type=str, default="noise",
    #                     help="type of modification, e.g., noise, scaling")
    args = parser.parse_args()
    
    if args.extract_layers:
        os.makedirs(args.feature_dir, exist_ok=True)
    
    set_seed(args.seed)
    eval_model(args)
