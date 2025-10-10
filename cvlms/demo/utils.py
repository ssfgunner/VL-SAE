import os
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import torch
import numpy as np

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Fail to Load Images {url}: {e}")

        placeholder = Image.new('RGB', (500, 350), color='lightgray')

        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(placeholder)
        draw.text((50, 175), "Fail to Load Images", fill="red")
        return placeholder

def display_concepts(concept_activations, concept2data, image_dir=None, top_k=5, display=3, vis_image=False, with_dataset=False):
    mean_activation = torch.Tensor([concept2data[str(i)]["mean_activation"] for i in range(len(concept2data))])
    mean_activation[mean_activation == 0] = float('inf')
    concept_activations = concept_activations / mean_activation
    top_k_vals, top_k_indices = torch.topk(concept_activations, top_k)
    for k, ind in enumerate(top_k_indices.squeeze()):

        item = concept2data[str(int(ind))]
        image_names = item["image_names"][:display]
        image_urls = item["image_urls"][:display]
        texts = list(set(item["texts"]))[:display]

        if vis_image:
            fig, axes = plt.subplots(ncols=display, figsize=(3*display, 3*display/3))
            if display == 1:
                axes = [axes]
            
            fig.suptitle(f"Images of Concept {ind}", fontsize=14)
            
            for i in range(display):
                if image_dir:
                    image = Image.open(os.path.join(image_dir, image_names[i]))
                else:
                    image = load_image_from_url(image_urls[i])                    
                image = image.resize((224, 224))
                axes[i].imshow(np.asarray(image))
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                # axes[i].set_title(f"Image {i+1}")
            plt.show()
        
        print("*"*40)
        print(f"Texts of Concept {ind}:")
        for i, text in enumerate(texts):
            print(f"Text {i}, {text}")  
        print("*"*40)
        