import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VL_SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, topk=32, dropout=0):
        super().__init__()
        self.encoder = nn.Parameter(torch.randn(hidden_dim, input_dim))
        nn.init.kaiming_uniform_(self.encoder, a=math.sqrt(5))
        
        self.vision_decoder = nn.Linear(hidden_dim, input_dim)
        self.text_decoder = nn.Linear(hidden_dim, input_dim)

        self.topk = topk
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def sparsify(self, embeddings, topk):
        abs_feat = torch.abs(embeddings)
        thres = torch.kthvalue(abs_feat, k=(self.hidden_dim - topk), dim=1)[0]
        sub = abs_feat - thres.unsqueeze(-1)
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        one_sub = torch.ones_like(n_sub)
        n_sub = torch.where(n_sub != 0, one_sub, n_sub)
        embeddings = embeddings * n_sub
        return embeddings

    def encode(self, embeddings, mode='eval'):
        weights = F.normalize(self.encoder, p=2, dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = torch.cdist(embeddings, weights, p=2)
        embeddings = 2 - embeddings
        return self.sparsify(embeddings, topk=self.topk)

    def forward(self, vision_embeddings=None, text_embeddings=None, mode='eval'):
        recon_vision_embeddings = None
        recon_text_embeddings = None
        latent_v = None
        latent_t = None
        if vision_embeddings is not None:
            latent_v = self.encode(vision_embeddings, mode=mode)
            recon_vision_embeddings = self.vision_decoder(latent_v)
        if text_embeddings is not None:
            latent_t = self.encode(text_embeddings, mode=mode)
            recon_text_embeddings = self.text_decoder(latent_t)
        return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t



class SAE_D(nn.Module):
    def __init__(self, input_dim, hidden_dim, topk=32, dropout=0.1):
        super().__init__()
        self.v_encoder = nn.Linear(input_dim, hidden_dim)
        self.activations = nn.ReLU()
        
        self.t_encoder = nn.Linear(input_dim, hidden_dim)
        self.vision_decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            )
        self.text_decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            )


        self.topk = topk
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def sparsify(self, embeddings):
        abs_feat = torch.abs(embeddings)
        thres = torch.kthvalue(abs_feat.float(), k=(self.hidden_dim - self.topk), dim=1)[0]

        sub = abs_feat - thres.unsqueeze(-1)
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        one_sub = torch.ones_like(n_sub)
        n_sub = torch.where(n_sub != 0, one_sub, n_sub)
        embeddings = embeddings * n_sub
   
        return embeddings
        

    def encode_v(self, embeddings):
        return self.sparsify(self.activations(self.v_encoder(embeddings)))
    
    def encode_t(self, embeddings):
        return self.sparsify(self.activations(self.t_encoder(embeddings)))

    def forward(self, vision_embeddings=None, text_embeddings=None):
        recon_vision_embeddings = None
        recon_text_embeddings = None
        latent_v = None
        latent_t = None
        if vision_embeddings is not None:
            latent_v = self.encode_v(vision_embeddings)
            recon_vision_embeddings = self.vision_decoder(latent_v)
        if text_embeddings is not None:
            latent_t = self.encode_t(text_embeddings)
            recon_text_embeddings = self.text_decoder(latent_t)
        return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t

class SAE_V(nn.Module):
    def __init__(self, input_dim, hidden_dim, topk=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.activations = nn.ReLU()
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            )

        self.topk = topk
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def sparsify(self, embeddings):
        abs_feat = torch.abs(embeddings)
        thres = torch.kthvalue(abs_feat.float(), k=(self.hidden_dim - self.topk), dim=1)[0]

        sub = abs_feat - thres.unsqueeze(-1)
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        one_sub = torch.ones_like(n_sub)
        n_sub = torch.where(n_sub != 0, one_sub, n_sub)
        embeddings = embeddings * n_sub
   
        return embeddings

    def encode(self, embeddings):
        return self.sparsify(self.activations(self.encoder(embeddings)))

    def forward(self, vision_embeddings=None, text_embeddings=None, mode='eval'):
        recon_vision_embeddings = None
        recon_text_embeddings = None
        latent_v = None
        latent_t = None
        if vision_embeddings is not None:
            latent_v = self.encode(vision_embeddings)
            recon_vision_embeddings = self.decoder(latent_v)
        if text_embeddings is not None:
            latent_t = self.encode(text_embeddings)
            recon_text_embeddings = self.decoder(latent_t)
        return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t