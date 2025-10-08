import sys
sys.path.append('../')
import torch
import os
import numpy as np
import argparse
import json
from sae_trainer.sae_model import VL_SAE, SAE_D, SAE_V, AuxiliaryAE
from tqdm import tqdm
from torch.cuda.amp import autocast

def parse_args():
    parser = argparse.ArgumentParser(description='save the concept interpretation for specific SAE weights')
    parser.add_argument('--topk', type=int, default=128, help='Top k concepts')
    parser.add_argument('--ckpt-path', type=str, default=None, help='Checkpoint path of SAE')
    parser.add_argument('--aux-ae-path', type=str, default=None, help='Checkpoint path of Auxiliary AE')
    parser.add_argument('--image-dir', type=str, default="../../CC3M/cc3m_jpg", help='Path to images')
    parser.add_argument('--embeddings_path', type=str, default="../representation_collection/activations/llava_cc3m_activations_model.layers.30_mean.pt", help='Path to embeddings')
    parser.add_argument('--hidden-ratio', type=int, default=8, help='Hidden dimension ratio')
    parser.add_argument('--input-dim', type=int, default=4096, help='Input dimension')
    parser.add_argument('--model-type', type=str, default='ViT-B-32', help='Model type')
    parser.add_argument('--sae-type', type=str, default='vlsae', help='SAE type')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for computation')
    parser.add_argument('--save-path', type=str, default='./concept2data', help='Path to save concept interpretations')
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(42)
    torch.manual_seed(42)

    hidden_dim = args.input_dim * args.hidden_ratio

    device = args.device

    alignment_model = AuxiliaryAE(vision_dim=args.input_dim, text_dim=args.input_dim).to(device)
    ckpt = torch.load(args.aux_ae_path, map_location='cpu')
    alignment_model.load_state_dict(ckpt)

    if args.sae_type == 'vlsae':
        autoencoder = VL_SAE(args.input_dim, hidden_dim, topk=args.topk).to(device)
    elif args.sae_type == 'saed':
        autoencoder = SAE_D(args.input_dim, hidden_dim, topk=args.topk).to(device)
    elif args.sae_type == 'saev':
        autoencoder = SAE_V(args.input_dim, hidden_dim, topk=args.topk).to(device)
        
    ckpt = torch.load(args.ckpt_path, map_location=device)
    autoencoder.load_state_dict(ckpt)

    # data loading
    embeddings_data = torch.load(args.embeddings_path, map_location='cpu')
    text_embeddings = torch.Tensor(np.stack(embeddings_data['text_features'], axis=0)).squeeze().half()
    image_embeddings = torch.Tensor(np.stack(embeddings_data['image_features'], axis=0)).squeeze().half()
    image_paths = embeddings_data['image_file']
    texts = embeddings_data['text']

    # deduplicate text embeddings and texts
    text_to_index = {text: idx for idx, text in enumerate(embeddings_data['text'])}
    texts = list(set(texts))
    text_indices = [text_to_index[text] for text in texts]
    text_embeddings = text_embeddings[text_indices]

    with open("../../CC3M/merged_cc3m.json", 'r') as f:
        cc3m_meta = json.load(f)
    image2url = {item['key']: item['url'] for item in cc3m_meta}


    def get_multiple_top_activations(target_indices, embeddings, references, top_k=10, batch_size=256, modality='vision'):
        # get top-k activations for multiple target indices

        all_target_activations = {idx: [] for idx in target_indices}
        
        with torch.no_grad():
            for i in tqdm(range(0, len(embeddings), batch_size), desc="Activation Collection"):
                batch_embeddings = embeddings[i:i + batch_size].to(device)
                # obtain activations for all targets [bs, num_targets]
                with autocast():
                    if modality == 'vision':
                        batch_embeddings, _, _, _ = alignment_model(vision_features=batch_embeddings)
                    else:
                        _, batch_embeddings, _, _ = alignment_model(text_features=batch_embeddings)
                    activations = autoencoder.encode(batch_embeddings)[:, target_indices]

                # storing the activations of each target separately
                for j, idx in enumerate(target_indices):
                    all_target_activations[idx].append(activations[:, j].cpu())
        
        results = {}
        mean_activations = {}
        # processing results for each target
        for target_idx in target_indices:
            target_activations = torch.cat(all_target_activations[target_idx])
            mean_activations[target_idx] = target_activations.mean().item()
            top_k_vals, top_k_indices = torch.topk(target_activations, top_k)
             # remove targets with all zero activations
            interpretation_data = []
            for val, idx in zip(top_k_vals, top_k_indices):
                reference = references[idx]
                interpretation_data.append(os.path.split(reference)[1])
            results[target_idx] = interpretation_data

            # if top_k_vals[0] > 0:
            #     interpretation_data = []
            #     for val, idx in zip(top_k_vals, top_k_indices):
            #         reference = references[idx]
            #         interpretation_data.append(os.path.split(reference)[1])
            #     results[target_idx] = interpretation_data
            # else:
            #     continue
        return results, mean_activations

    target_indices = range(args.input_dim*args.hidden_ratio)



    concept2data = {}

    concept_batch = 1000  # process 50 concepts at a time to save memory
    num_concept_batches = (len(target_indices) + concept_batch - 1) // concept_batch
    for batch_idx in range(num_concept_batches):
        batch_start = batch_idx * concept_batch
        batch_end = min((batch_idx + 1) * concept_batch, len(target_indices))
        target_indices_batch = list(range(batch_start, batch_end))
        image_results, image_mean_activations = get_multiple_top_activations(target_indices_batch, image_embeddings, image_paths, modality='vision', batch_size=2048)
        text_results, text_mean_activations = get_multiple_top_activations(target_indices_batch, text_embeddings, texts, modality='text', batch_size=2048)

        for target_idx in tqdm(target_indices_batch, desc="Saving results"):
            image_url_list = [image2url[image_name.split('.')[0]] for image_name in image_results[target_idx]]
            text_list = text_results[target_idx]
            mean_activation = (image_mean_activations[target_idx] + text_mean_activations[target_idx]) / 2.0            
            concept2data[target_idx] = {
                'image_urls': image_url_list,
                'image_names': image_results[target_idx],
                'texts': text_list,
                'mean_activation': float(mean_activation)
            }

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    with open(os.path.join(args.save_path, f'c2d_llava_{args.topk}_{args.hidden_ratio}.json'), 'w') as f:
        json.dump(concept2data, f, indent=4)

if __name__ == '__main__':
    main()
