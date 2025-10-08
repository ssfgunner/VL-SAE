import sys
sys.path.append('../')
import torch
import torch.nn as nn
import open_clip
from sae_trainer.sae_model import VL_SAE


class OPENCLIP_VLSAE(nn.Module):
    def __init__(self, model_name, pretrained, hidden_ratio=32, topk=32, input_dim=512, sae_path=None):
        super(OPENCLIP_VLSAE, self).__init__()
        self.clip_model, _, self.transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        
        self.input_dim = input_dim
        self.hidden_dim = input_dim * hidden_ratio
        self.topk = topk
        
        self.sae = None

        if sae_path is not None:
            self.sae = VL_SAE(self.input_dim, self.hidden_dim, self.topk)
            state_dict = torch.load(sae_path, map_location='cpu')
            self.sae.load_state_dict(state_dict)
            print(f"Successfully loaded pretrained model {sae_path}.")
        
    def encode_image(self, x):
        embeddings = self.clip_model.encode_image(x)
        if self.sae is not None:
            embeddings = self.sae(vision_embeddings=embeddings)[0]
        return embeddings
        
    def encode_text(self, x):
        embeddings = self.clip_model.encode_text(x)
        if self.sae is not None:
            embeddings = self.sae(text_embeddings=embeddings)[1]
        return embeddings

    def encode_text_enhance(self, x):
        embeddings = self.clip_model.encode_text(x)
        assert self.sae is not None
        concepts = self.sae(text_embeddings=embeddings)[1]
        assert len(concepts.shape) == 2
        return embeddings, concepts

    def encode_image_enhance(self, x):
        embeddings = self.clip_model.encode_image(x)
        assert self.sae is not None
        concepts = self.sae(vision_embeddings=embeddings)[0]
        assert len(concepts.shape) == 2
        return embeddings, concepts

    def encode_text_concept(self, x):
        embeddings = self.clip_model.encode_text(x)
        assert self.sae is not None
        concepts = self.sae.encode(embeddings)
        assert len(concepts.shape) == 2
        return embeddings, concepts

    def encode_image_concept(self, x):
        embeddings = self.clip_model.encode_image(x)
        assert self.sae is not None
        concepts = self.sae.encode(embeddings)
        assert len(concepts.shape) == 2
        return embeddings, concepts

def load_open_clip_vlsae(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu", sae_path=None, hidden_ratio=32, topk=32, input_dim=768, **kwargs):
    _, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir)
    model = OPENCLIP_VLSAE(model_name, pretrained, sae_path=sae_path, hidden_ratio=hidden_ratio, topk=topk, input_dim=input_dim)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer