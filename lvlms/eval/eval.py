import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DirectoryAnalyzer:
    def __init__(self):

        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14', force_download=False)
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14', force_download=False).to(device)
        
    def extract_features(self, target_dir):
        dirs = [os.path.join(target_dir, d) for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
        all_clip_image_features = []
        all_clip_text_features = []
        
        for dir_path in dirs:
            images = [f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

            with open(os.path.join(dir_path, 'text_interpretation.txt'), 'r') as f:
                # text = [item.strip() for item in f.readlines()]
                text = [item for item in f.readlines()]
            with torch.no_grad():
                text_inputs = self.clip_processor(text=text, truncation=True, return_tensors="pt", padding=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                clip_text_features = self.clip_model.get_text_features(**text_inputs).cpu()
            clip_img_features = []
            for img_file in images:
                image = Image.open(os.path.join(dir_path, img_file))
                with torch.no_grad():
                    image_inputs = self.clip_processor(images=image, return_tensors="pt")
                    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                    clip_img_features.append(self.clip_model.get_image_features(**image_inputs).squeeze().cpu())
            clip_img_features = torch.stack(clip_img_features)
            
            all_clip_image_features.append(clip_img_features)
            all_clip_text_features.append(clip_text_features)
                    
        return all_clip_image_features, all_clip_text_features, dirs

    def compute_similarities(self, features1, features2=None):

        if features2 is None:
            features2 = features1

        features1 = features1.mean(dim=0, keepdim=True)
        features2 = features2.mean(dim=0, keepdim=True)
        features1_norm = F.normalize(features1, p=2, dim=-1)
        features2_norm = F.normalize(features2, p=2, dim=-1)
        return features1_norm @ features2_norm.T
    
    def analyze_directories(self, target_dir):
        clip_img_features, clip_txt_features, dirs = self.extract_features(target_dir)
        
        clip_intra_sim = 0
        for i in range(len(clip_img_features)):
            clip_intra_sim += self.compute_similarities(clip_img_features[i], clip_txt_features[i]).max()
        clip_intra_sim /= len(clip_img_features)
        print("Intra similarity", clip_intra_sim)

        clip_inter_sim = 0
        for i in range(len(clip_img_features)):
            for j in range(i + 1, len(clip_txt_features)):
                clip_inter_sim += self.compute_similarities(clip_img_features[i], clip_txt_features[j]).max()
        clip_inter_sim /= (len(clip_img_features) * (len(clip_txt_features) - 1) / 2)
        print("Inter similarity", clip_inter_sim)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='analyse directories of images and text interpretations')
    parser.add_argument('--target-dir', type=str, default=None, required=True,
                        help='includes multiple subdirectories, each with images and a text_interpretation.txt file')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='使用的设备 (默认: cuda:0)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    analyzer = DirectoryAnalyzer()
    analyzer.analyze_directories(args.target_dir)