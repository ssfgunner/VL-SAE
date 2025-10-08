import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sae_model import VL_SAE, SAE_D, SAE_V

def get_args_parser():
    parser = argparse.ArgumentParser('Train VL-SAE with OpenCLIP Embeddings', add_help=False)
    parser.add_argument('--pretrained_model', default="ViT-B-32", type=str)
    parser.add_argument('--text_embeddings_path', default=None, type=str)
    parser.add_argument('--image_embeddings_path', default=None, type=str)
    parser.add_argument('--model_type', default="VL_SAE", type=str)
    parser.add_argument('--save_path', default="./", type=str)
    parser.add_argument('--topk', default=128, type=int)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--save_final', action='store_true')
    parser.add_argument('--training_ratio', default=0.8, type=float)
    parser.add_argument('--hidden_ratio', default=32, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--warmup_epochs', default=2, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--initial_lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--patience', default=10, type=int)

    return parser

# topk = 128
# hidden_ratio = 32
def main(args):
    device = 'cuda:0'
    clip_text_embeddings = torch.load(args.text_embeddings_path, map_location='cpu')['embeddings']
    clip_image_embeddings = torch.load(args.image_embeddings_path, map_location='cpu')['embeddings']

    input_dim = clip_text_embeddings.shape[1]
    hidden_dim = input_dim * args.hidden_ratio
    num_epochs = args.num_epochs
    warmup_epochs = args.warmup_epochs
    batch_size = args.batch_size
    initial_lr = args.initial_lr
    topk = args.topk
    hidden_ratio = args.hidden_ratio
    

    weight_decay = args.weight_decay
    patience = args.patience
    if args.model_type == 'VL_SAE':
        autoencoder = VL_SAE(input_dim, hidden_dim, topk=args.topk, dropout=0).to(device)
    elif args.model_type == 'SAE_D':
        autoencoder = SAE_D(input_dim, hidden_dim, topk=args.topk, dropout=0).to(device)
    elif args.model_type == 'SAE_V':
        autoencoder = SAE_V(input_dim, hidden_dim, topk=args.topk, dropout=0).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)

    total_samples = len(clip_text_embeddings)
    train_ratio = args.training_ratio
    indices = np.random.permutation(total_samples)
    train_size = int(total_samples * train_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_text_embeddings = clip_text_embeddings[train_indices]
    train_image_embeddings = clip_image_embeddings[train_indices]
    val_text_embeddings = clip_text_embeddings[val_indices]
    val_image_embeddings = clip_image_embeddings[val_indices]

    if args.shuffle:
        print("Shuffling the training and validation data...")

        rand_indx = torch.randperm(len(train_text_embeddings))
        train_text_embeddings = train_text_embeddings[rand_indx]

        rand_indx = torch.randperm(len(train_image_embeddings))
        train_image_embeddings = train_image_embeddings[rand_indx]

        rand_indx = torch.randperm(len(val_text_embeddings))
        val_text_embeddings = val_text_embeddings[rand_indx]

        rand_indx = torch.randperm(len(val_image_embeddings))
        val_image_embeddings = val_image_embeddings[rand_indx]

    best_val_loss = float('inf')
    patience_counter = 0
    autoencoder.train()

    num_steps = len(train_text_embeddings) // batch_size
    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            lr = initial_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()

        epoch_loss = 0
        for i in tqdm(range(0, len(train_text_embeddings), batch_size)):
            optimizer.zero_grad()

            batch_embeddings_v = train_image_embeddings[i:i + batch_size].to(device)
            batch_embeddings_t = train_text_embeddings[i:i + batch_size].to(device)
            recon_v, recon_t, _, _ = autoencoder(batch_embeddings_v, batch_embeddings_t)        
            recon_loss = criterion(recon_v, batch_embeddings_v) + criterion(recon_t, batch_embeddings_t)
            recon_loss.backward()
            optimizer.step()
            epoch_loss += recon_loss.item()

        val_loss = validate(autoencoder, val_text_embeddings, val_image_embeddings, 
                        criterion, batch_size, device)
        
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], LR: {lr}, '
            f'Train Loss: {epoch_loss/num_steps:.4f}, '
            f'Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(autoencoder.state_dict(), 
                    os.path.join(args.save_path, f'openclip_{args.pretrained_model}_{args.model_type}_{topk}_{hidden_ratio}_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch + 1}')
                break
    if args.save_final:
        torch.save(autoencoder.state_dict(), 
                os.path.join(args.save_path, f'openclip_{args.pretrained_model}_{args.model_type}_{topk}_{hidden_ratio}_{best_val_loss}_final.pth'))

def validate(model, val_text_embeddings, val_image_embeddings, criterion, batch_size, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, len(val_text_embeddings), batch_size):
            batch_embeddings_v = val_image_embeddings[i:i+batch_size].to(device)
            batch_embeddings_t = val_text_embeddings[i:i+batch_size].to(device)
            recon_v, recon_t, _, _ = model(batch_embeddings_v, batch_embeddings_t)
            loss = criterion(recon_v, batch_embeddings_v) + criterion(recon_t, batch_embeddings_t)
            total_loss += loss.item() * batch_embeddings_v.size(0)
    avg_loss = total_loss / len(val_text_embeddings)
    model.train()
    return avg_loss

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    main(args)    
