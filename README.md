# [NeurIPS 2025] VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set

This repository is the official implementation of [VL-SAE](https://arxiv.org/abs/2030.12345), which helps users to understand the vision-language alignment of VLMs via concepts.

## Requirements

Create a conda virtual environment and activate it:

```bash
conda create -n vlsae python=3.8 -y
conda activate vlsae
```

Install dependencies:

```
pip install -r requirements.txt
```

## Dataset preparation

Install CC3M dataset from [cc3m-wds](https://huggingface.co/datasets/pixparse/cc3m-wds), and put it under ``./CC3M``

Running the provided scripts to preprocess the dataset:

``` bash
bash cc3m_untar.sh
python cc3m_moving.py
python cc3m_meta.py
```

## Quick Start

Download the pre-trained weights and put it under 

Download the metadata of pre-trained SAE and put it under

We present the inference demo of VL-SAE with OpenCLIP and LLaVA 1.5 in ``./cvlms/demo/demo.ipynb`` and ``./lvlms/demo/demo.ipynb``, respectively.

## Training

This repo supports the construction of VL-SAE for [LLaVA-1.5](https://huggingface.co/liuhaotian/llava-v1.5-7b) and [OpenCLIP](https://www.google.com/search?client=safari&rls=en&q=openclip+github&ie=UTF-8&oe=UTF-8).

First, collect the hidden representations of pre-trained models:

```bash
model_type="cvlms" # for OpenCLIP
# model_type="lvlms" # for LLaVA
cd ./${model_type}/representation_collection
bash get_activations.sh
```

With a single NVIDIA RTX 4090, this step takes approximately 5 hours for OpenCLIP and 4 days for LLaVA. If you wish to skip this step, you can download the pre-computed features of [OpenCLIP]() and [LLaVA](), then place them under `VL-SAE/cvlms/representation_collection/activations` and `VL-SAE/lvlms/representation_collection/activations`, respectively.

Then, train VL-SAE based on the collected representations:

```bash
cd ../sae_trainer
bash train.sh
```

## Evaluation

### Qualitative Results

Visualize the concepts learned by VL-SAE:

```bash
cd ../eval
python visualize_concept.py --topk 256 --ckpt-path ../sae_trainer/sae_weights/openclip_ViT-B-32_VL_SAE_256_8_best.pth
```

Each concept is represented by a set of images stored in the corresponding folder and sentences in the `text_interpretation.txt` file. 

![concept_vis](./figure/concept_vis.png)

### Quantitive Results

With the visualizations of concepts, their inter-similarity score and intra-similarity score can be computed using CLIP embeddings.

```bash
python eval.py --target-dir ./concept_images/vlsae_ViT-B-32_256
```

![concept_eval](./figure/concept_eval.png)

### Generate Metadata Files for SAE Weights

Finally, generate a JSON file for the trained SAE, which stores the index of each concept, along with its mean activation value, and maximum activation data (image URL, texts). This file is designed to support the integration of VL-SAE into the model inference process for interpretability purposes.

```bash
python concept2data.py --topk 256 --ckpt-path ../sae_trainer/sae_weights/openclip_ViT-B-32_VL_SAE_256_8_best.pth
```

## Pre-trained Models

| Base Model        | Top-K | Hidden-Ratio | Download          |
| ----------------- | ----- | ------------ | ----------------- |
| OpenCLIP-ViT-B/32 | 256   | 8            | [Model-Weights]() |
| OpenCLIP-ViT-B/16 | 256   | 8            | [Model-Weights]() |
| OpenCLIP-ViT-L/14 | 256   | 8            | [Model-Weights]() |
| OpenCLIP-ViT-H/14 | 256   | 8            | [Model-Weights]() |
| LLaVA-1.5-7B      | 256   | 8            | [Model-Weights]() |

## Enhancing Vision-Language Alignment

### Zero-shot Image Classification



### Hallucination Elimination



## Citation

If you find VL-SAE useful for your research and applications, please cite using this BibTeX:

```latex
@misc{shen2025vlsae,
      title={VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set}, 
      author={Shen, Shufan and Sun, Junshu, and Huang, Qingming and Wang, Shuhui},
      publisher={NeurIPS},
      year={2025},
}
```

## Related Projects

[OpenCLIP]() 

[VCD]()

