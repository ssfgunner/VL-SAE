import sys
sys.path.append('../')
import torch
import os
import numpy as np
import argparse
from sae_trainer.sae_model import VL_SAE, SAE_D, SAE_V, AuxiliaryAE
from tqdm import tqdm
from torch.cuda.amp import autocast

def parse_args():
    parser = argparse.ArgumentParser(description='VLSAE interpreter')
    parser.add_argument('--topk', type=int, default=128, help='Top k concepts')
    parser.add_argument('--ckpt-path', type=str, default=None, help='Checkpoint path of SAE')
    parser.add_argument('--aux-ae-path', type=str, default=None, help='Checkpoint path of Auxiliary AE')
    parser.add_argument('--image-dir', type=str, default="../../CC3M/cc3m_jpg", help='Path to images')
    parser.add_argument('--embeddings_path', type=str, default="../representation_collection/activations/llava_cc3m_activations_model.layers.30_mean.pt", help='Path to embeddings')
    parser.add_argument('--hidden-ratio', type=int, default=8, help='Hidden dimension ratio')
    parser.add_argument('--input-dim', type=int, default=4096, help='Input dimension')
    parser.add_argument('--sae-type', type=str, default='vlsae', help='SAE type')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for computation')
    parser.add_argument('--save-path', type=str, default='./concept_images', help='Path to save concept interpretations')
    parser.add_argument('--num-targets', type=int, default=100, help='Number of target concepts to interpret')
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
        # processing results for each target
        for target_idx in target_indices:
            target_activations = torch.cat(all_target_activations[target_idx])
            top_k_vals, top_k_indices = torch.topk(target_activations, top_k)
            # remove targets with all zero activations
            if top_k_vals[0] > 0:
                interpretation_data = []
                for val, idx in zip(top_k_vals, top_k_indices):
                    reference = references[idx]
                    interpretation_data.append(os.path.split(reference)[1])
                results[target_idx] = interpretation_data
            else:
                continue        
        return results

    

    target_indices = np.random.choice(args.input_dim*args.hidden_ratio, size=args.num_targets, replace=False)

    image_results = get_multiple_top_activations(target_indices, image_embeddings, image_paths, modality='vision')
    text_results = get_multiple_top_activations(target_indices, text_embeddings, texts, modality='text')

    for target_idx in tqdm(target_indices, desc="Saving results"):
        if target_idx not in image_results.keys() or target_idx not in text_results.keys():
            continue
        image_save_dir = os.path.join(args.save_path, f'{args.sae_type}_llava_{args.topk}', str(target_idx))
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir, exist_ok=True)
        
        for image_name in image_results[target_idx]:
            image_path = os.path.join(args.image_dir, image_name)
            image_save_path = os.path.join(image_save_dir, image_name)
            os.system(f'cp {image_path} {image_save_path}')
        
        with open(os.path.join(image_save_dir, 'text_interpretation.txt'), 'w') as f:
            f.write('\n'.join(text_results[target_idx]))

if __name__ == '__main__':
    main()
