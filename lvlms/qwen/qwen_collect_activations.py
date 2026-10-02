#!/usr/bin/env python
"""Collect per-layer vision/text hidden features from Qwen2.5-VL-3B for VL-SAE training.

Input: a pairs jsonl with {"image": <path relative to --image-root>, "text": <caption>}.
For each pair runs ONE prompt-only forward with the image:
  - vision features (mean over <|image_pad|> span)
  - text features   (mean over non-special text tokens)

Saves one .pt per (layer, shard):
  {"image_features": [...], "text_features": [...],
   "image_file": [...], "text": [...]}   (lists of fp16 arrays / str)

Usage:
  python qwen_collect_activations.py --shard 0 --num-shards 4 --batch-size 16
"""
import argparse
import json
import os
import re

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_PATH = os.environ.get("QWEN_VL_MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")
IMG_TOK = 151655  # <|image_pad|>
SPECIAL = frozenset({151644, 151645, 151652, 151653, 151655})  # im_start/im_end/vision_start/vision_end/image_pad


def find_decoder_layers(model):
    """idx -> module for the text decoder layers (excludes visual encoder)."""
    layers = {}
    for name, mod in model.named_modules():
        m = re.search(r"layers\.(\d+)$", name)
        if m and "visual" not in name:
            layers[int(m.group(1))] = mod
    return layers


def pool_features(hidden, input_ids, attention_mask):
    """hidden [B,T,D] -> (vision_feats [B,D], text_feats [B,D]) masked mean-pools."""
    img_mask = input_ids == IMG_TOK
    special = torch.isin(input_ids, torch.tensor(list(SPECIAL), device=input_ids.device))
    text_mask = attention_mask.bool() & ~special & ~img_mask
    h = hidden.float()
    vf = (h * img_mask.unsqueeze(-1)).sum(1) / img_mask.sum(1, keepdim=True).clamp(min=1)
    tf = (h * text_mask.unsqueeze(-1)).sum(1) / text_mask.sum(1, keepdim=True).clamp(min=1)
    return vf.half().cpu(), tf.half().cpu()


def run_batch(model, processor, hooks_out, images, texts):
    convs = [[{"role": "user", "content": [{"type": "image", "image": im},
                                           {"type": "text", "text": tx}]}]
             for im, tx in zip(images, texts)]
    prompts = [processor.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
               for c in convs]
    inputs = processor(text=prompts, images=images,
                       padding=True, return_tensors="pt").to(model.device)
    hooks_out.clear()
    with torch.inference_mode():
        model(**inputs, use_cache=False)
    return inputs, dict(hooks_out)


def main():
    ap = argparse.ArgumentParser()
    Q = os.environ.get(
        "VLSAE_QWEN_WORKDIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_workdir"),
    )
    ap.add_argument("--pairs", default=os.path.join(Q, "cc3m_pairs.jsonl"))
    ap.add_argument("--image-root", default=Q)
    ap.add_argument("--out-dir", default=os.path.join(Q, "activations"))
    ap.add_argument("--layers", default="26")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first N pairs of this shard")
    ap.add_argument("--save-every", type=int, default=500, help="checkpoint every N batches")
    args = ap.parse_args()
    layer_idxs = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    pairs = []
    with open(args.pairs) as f:
        for line in f:
            p = json.loads(line)
            if p.get("text", "").strip():
                pairs.append(p)
    pairs = [p for i, p in enumerate(pairs) if i % args.num_shards == args.shard]
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[collect] shard {args.shard}/{args.num_shards}: {len(pairs)} pairs, layers {layer_idxs}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    dec_layers = find_decoder_layers(model)
    assert len(dec_layers) == 36, f"expected 36 decoder layers, found {len(dec_layers)}"

    hooks_out = {}
    handles = []

    def make_hook(idx):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hooks_out[idx] = h
        return hook

    for idx in layer_idxs:
        handles.append(dec_layers[idx].register_forward_hook(make_hook(idx)))

    store = {l: {"image_features": [], "text_features": [],
                 "image_file": [], "text": []} for l in layer_idxs}

    def save(tag=""):
        for l in layer_idxs:
            path = os.path.join(args.out_dir, f"qwen_cc3m_layer{l}_shard{args.shard}{tag}.pt")
            torch.save(store[l], path + ".tmp")
            os.replace(path + ".tmp", path)

    bs = args.batch_size
    n_batches = (len(pairs) + bs - 1) // bs
    n_skip = 0
    for bi in range(n_batches):
        chunk = pairs[bi * bs:(bi + 1) * bs]
        images, texts, keep = [], [], []
        for p in chunk:
            try:
                images.append(Image.open(os.path.join(args.image_root, p["image"])).convert("RGB"))
                texts.append(p["text"])
                keep.append(p)
            except Exception as ex:
                n_skip += 1
                if n_skip <= 5:
                    print(f"[collect] skip {p['image']}: {ex}", flush=True)
        if not images:
            continue
        inputs_img, hid_img = run_batch(model, processor, hooks_out, images, texts)
        for l in layer_idxs:
            vf, tf = pool_features(hid_img[l], inputs_img["input_ids"], inputs_img["attention_mask"])
            store[l]["image_features"].extend(vf.numpy())
            store[l]["text_features"].extend(tf.numpy())
            store[l]["image_file"].extend([p["image"] for p in keep])
            store[l]["text"].extend(texts)
        del inputs_img, hid_img
        if (bi + 1) % 50 == 0 or bi + 1 == n_batches:
            print(f"[collect] shard {args.shard} batch {bi + 1}/{n_batches} skipped={n_skip}", flush=True)
        if (bi + 1) % args.save_every == 0:
            save("_ckpt")
    save()
    for h in handles:
        h.remove()
    print(f"[collect] DONE shard {args.shard}: {len(store[layer_idxs[0]]['text'])} pairs saved, {n_skip} skipped", flush=True)


if __name__ == "__main__":
    main()
