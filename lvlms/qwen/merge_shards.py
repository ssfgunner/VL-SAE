#!/usr/bin/env python
"""Merge per-shard activation files into one training file per layer (node side).

Input:  activations/qwen_cc3m_layer{L}_shard{k}.pt   (k = 0..num_shards-1)
Output: activations/qwen_cc3m_layer{L}_merged.pt
"""
import argparse
import glob
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir", default=None)
    ap.add_argument("--layers", default="26")
    args = ap.parse_args()
    act_dir = args.act_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "activations")

    for layer in [int(x) for x in args.layers.split(",")]:
        merged = {"image_features": [], "text_features": [],
                  "image_file": [], "text": []}
        files = sorted(glob.glob(os.path.join(act_dir, f"qwen_cc3m_layer{layer}_shard[0-9]*.pt")))
        files = [f for f in files if "_ckpt" not in f]
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            for k in merged:
                merged[k].extend(d[k])
        out = os.path.join(act_dir, f"qwen_cc3m_layer{layer}_merged.pt")
        torch.save(merged, out)
        print(f"layer {layer}: {len(merged['text'])} pairs from {len(files)} shards -> {out}", flush=True)


if __name__ == "__main__":
    main()
