import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.cuda.amp import autocast
from sae_model import AuxiliaryAE
import os

def get_args_parser():
    parser = argparse.ArgumentParser('Train Auxiliary AE with LLaVA Embeddings', add_help=False)
    parser.add_argument('--embeddings_path', default=None, type=str)
    parser.add_argument('--save_path', default="./", type=str)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--save_final', action='store_true')
    parser.add_argument('--training_ratio', default=0.8, type=float)
    parser.add_argument('--hidden_ratio', default=8, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--warmup_steps', default=100, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--initial_lr', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--temperature', default=0.07, type=float)

    return parser

def contrastive_loss(vision_embed, text_embed, temperature=0.07):
    #vision_embed = F.normalize(vision_embed, dim=-1)
    #text_embed = F.normalize(text_embed, dim=-1)
    sim_matrix = torch.matmul(vision_embed, text_embed.transpose(0, 1)) / temperature    
    labels = torch.arange(vision_embed.shape[0], device=vision_embed.device)
    loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.transpose(0, 1), labels)
    return loss / 2.0

class AlignmentTrainer:
    def __init__(self, args, vision_dim, text_dim, config=None):
        self.config = {
            'lr': args.initial_lr,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'warmup_steps': args.warmup_steps,
            'weight_decay': args.weight_decay,
            'temperature': args.temperature,
        }
        if config:
            self.config.update(config)
            
        self.model = AuxiliaryAE(vision_dim, text_dim).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['num_epochs']
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_contrast_loss = 0
        total_recon_loss = 0
        
        for batch_idx, (vision_features, text_features) in enumerate(train_loader):
            self.optimizer.zero_grad()
            with autocast():
                vision_embed, text_embed, vision_recon, text_recon = self.model(vision_features.to(device), text_features.to(device))
            
            contrast_loss = contrastive_loss(vision_embed, text_embed, self.config['temperature'])
            vision_recon_loss = nn.MSELoss()(vision_recon, vision_features.to(device))
            text_recon_loss = nn.MSELoss()(text_recon, text_features.to(device))
            recon_loss = vision_recon_loss + text_recon_loss
            loss = 1*contrast_loss + recon_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_contrast_loss += contrast_loss.item()
            total_recon_loss += recon_loss.item()
            
        return {
            'total_loss': total_loss / len(train_loader),
            'contrast_loss': total_contrast_loss / len(train_loader),
            'recon_loss': total_recon_loss / len(train_loader)
        }
    
        
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_contrast_loss = 0
        total_recon_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (val_vision_features, val_text_features) in enumerate(val_loader):
                with autocast():
                    vision_embed, text_embed, vision_recon, text_recon = self.model(
                        val_vision_features.to(device), 
                        val_text_features.to(device)
                    )
                
                contrast_loss = contrastive_loss(vision_embed, text_embed, self.config['temperature'])
                vision_recon_loss = nn.MSELoss()(vision_recon, val_vision_features.to(device))
                text_recon_loss = nn.MSELoss()(text_recon, val_text_features.to(device))
                recon_loss = vision_recon_loss + text_recon_loss
                loss = 1*contrast_loss + recon_loss
                
                similarity = torch.matmul(vision_embed, text_embed.transpose(0, 1))
                predictions = similarity.argmax(dim=-1)
                labels = torch.arange(len(predictions), device=predictions.device)
                
                total_loss += loss.item() * len(predictions)
                total_contrast_loss += contrast_loss.item() * len(predictions)
                total_recon_loss += recon_loss.item() * len(predictions)
                total_correct += (predictions == labels).sum().item()
                total_samples += len(predictions)
        
        return {
            'avg_loss': total_loss / total_samples,
            'contrast_loss': total_contrast_loss / total_samples,
            'recon_loss': total_recon_loss / total_samples,
            'accuracy': total_correct / total_samples
        }

if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = {
        'lr': args.initial_lr,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'temperature': args.temperature,
    }

    trainer = AlignmentTrainer(args, vision_dim=4096, text_dim=4096, config=config)
    trainer.model = trainer.model.to(device)

    embeddings_data = torch.load(args.embeddings_path)

    text_embeddings = torch.Tensor(np.stack(embeddings_data['text_features'], axis=0)).squeeze().half()
    image_embeddings = torch.Tensor(np.stack(embeddings_data['image_features'], axis=0)).squeeze().half()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    total_samples = len(text_embeddings)
    train_ratio = 0.8
    indices = np.random.permutation(total_samples)
    train_size = int(total_samples * train_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_text_embeddings = text_embeddings[train_indices]
    train_image_embeddings = image_embeddings[train_indices]
    val_text_embeddings = text_embeddings[val_indices]
    val_image_embeddings = image_embeddings[val_indices]

    train_dataset = TensorDataset(train_image_embeddings, train_text_embeddings)
    val_dataset = TensorDataset(val_image_embeddings, val_text_embeddings)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_metrics = trainer.train_epoch(train_loader)
        
        val_metrics = trainer.evaluate(val_loader)
        
        trainer.scheduler.step()
        
        print(f"\nEpoch [{epoch+1}/{config['num_epochs']}]")
        print(f"Training Metrics:")
        print(f"  Total Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Contrast Loss: {train_metrics['contrast_loss']:.4f}")
        print(f"  Recon Loss: {train_metrics['recon_loss']:.4f}")
        print(f"Validation Metrics:")
        print(f"  Total Loss: {val_metrics['avg_loss']:.4f}")
        print(f"  Contrast Loss: {val_metrics['contrast_loss']:.4f}")
        print(f"  Recon Loss: {val_metrics['recon_loss']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']*100:.2f}%")
        
        if val_metrics['avg_loss'] < best_val_loss:
            best_val_loss = val_metrics['avg_loss']
            torch.save(trainer.model.state_dict(), os.path.join(args.save_path, './llava_aux_best.pt'))
        
    if args.save_final:
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }, os.path.join(args.save_path, './llava_aux_last.pt'))