import os
import os.path as osp
import argparse
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image

import open_clip

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        return self.processor(image), self.image_paths[idx]


def get_args_parser():
    parser = argparse.ArgumentParser('Obtain features from open-clip', add_help=False)
    parser.add_argument('--pretrained_model', default="ViT-B-32", type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--text_path', default=None, type=str)
    parser.add_argument('--image_path', default=None, type=str)
    parser.add_argument('--save_path', default="./", type=str)

    return parser

def get_text_embeddings(model, tokenizer, texts, batch_size=128):
    model = model.cuda()
    
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Text Embedding Extraction'):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts).cuda()
        
        with torch.no_grad():
            embeddings = model.encode_text(inputs).cpu()
            all_embeddings.append(embeddings)
    return torch.cat(all_embeddings)

def get_image_embeddings(model, processor, image_paths, batch_size=256):
    model = model.cuda()
    all_embeddings = []
    dataset = ImageDataset(image_paths, processor)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=4,
        pin_memory=True
    )
    total_paths = []
    for batch_images, batch_paths in tqdm(dataloader, desc='Image Embedding Extraction'):
        inputs = batch_images.cuda()
        total_paths += batch_paths
        with torch.no_grad():
            embeddings = model.encode_image(inputs).cpu()
            all_embeddings.append(embeddings)
    
    return torch.cat(all_embeddings), total_paths

def main(args):
    # load pre-trained models
    model, _, processor = open_clip.create_model_and_transforms(args.pretrained_model, pretrained=args.model_path)
    model.eval() 
    tokenizer = open_clip.get_tokenizer(args.pretrained_model)
    if not osp.exists(args.save_path):
        os.mkdir(args.save_path)

    # obtain text embeddings
    if args.text_path is not None:
        # with open(args.text_path, 'r') as f:
        #     texts = [line.strip() for line in f.readlines()]
        with open(args.text_path, "r") as f:
            datasets = json.load(f)
        texts = [item['caption'] for item in datasets]
        text_embeddings = get_text_embeddings(model, tokenizer, texts)
        save_dict = {
            'embeddings': text_embeddings,
            'texts': texts
        }
        torch.save(save_dict, osp.join(args.save_path, f"{args.pretrained_model}_text_embeddings.pt"))
    
    if args.image_path is not None:
        image_paths = [os.path.join(args.image_path, img) for img in os.listdir(args.image_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        image_embeddings, total_paths = get_image_embeddings(model, processor, image_paths)
        save_dict = {
            'embeddings': image_embeddings,
            'image_paths': total_paths
        }
        torch.save(save_dict, osp.join(args.save_path, f"{args.pretrained_model}_vision_embeddings.pt"))

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)    
