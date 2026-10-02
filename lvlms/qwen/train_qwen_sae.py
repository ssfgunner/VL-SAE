#!/usr/bin/env python
"""Two-stage VL-SAE training for one Qwen2.5-VL-3B decoder layer.

Stage 1: AuxiliaryAE (contrastive + recon) aligns vision/text features.
Stage 2: VL_SAE (TopK) trained on aux-encoded embeddings.

Usage:
  python train_qwen_sae.py --layer 26 --device cuda:0
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# AuxiliaryAE / VL_SAE come from the upstream trainer (flat repo layout,
# no package __init__ — same import style as lvlms/sae_trainer/train.py).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sae_trainer"))
from sae_model import VL_SAE, AuxiliaryAE  # noqa: E402


def contrastive_loss(vision_embed, text_embed, temperature=0.07):
    sim = torch.matmul(vision_embed, text_embed.transpose(0, 1)) / temperature
    labels = torch.arange(vision_embed.shape[0], device=vision_embed.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.transpose(0, 1), labels)) / 2.0


def load_features(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    vf = torch.from_numpy(np.stack(d["image_features"])).float()
    tf = torch.from_numpy(np.stack(d["text_features"])).float()
    return vf, tf


def split(n, ratio=0.9, seed=42):
    idx = np.random.RandomState(seed).permutation(n)
    cut = int(n * ratio)
    return idx[:cut], idx[cut:]


def train_aux(vf, tf, args, device, out_dir):
    # projection_dim = feature dim (2048 for Qwen2.5-VL-3B); the upstream
    # default of 4096 matches their LLaVA hidden size, not ours.
    aux = AuxiliaryAE(vf.shape[1], tf.shape[1], projection_dim=vf.shape[1]).to(device)
    opt = torch.optim.AdamW(aux.parameters(), lr=args.aux_lr, weight_decay=0.01)
    tr, va = split(len(vf)); va = va[:20480]  # cap val: full-val contrastive is O(n^2)
    best = float("inf")
    for epoch in range(args.aux_epochs):
        aux.train()
        perm = torch.randperm(len(tr))
        tot = 0.0
        for i in range(0, len(perm), args.aux_batch):
            b = perm[i:i + args.aux_batch]
            bv, bt = vf[tr[b]].to(device), tf[tr[b]].to(device)
            opt.zero_grad()
            ve, te, vr, tr_ = aux(bv, bt)
            loss = contrastive_loss(ve, te) + nn.MSELoss()(vr, bv) + nn.MSELoss()(tr_, bt)
            loss.backward()
            opt.step()
            tot += loss.item()
        aux.eval()
        with torch.no_grad():
            ve, te, vr, tr_ = aux(vf[va].to(device), tf[va].to(device))
            vloss = (contrastive_loss(ve, te) + nn.MSELoss()(vr, vf[va].to(device))
                     + nn.MSELoss()(tr_, tf[va].to(device))).item()
        print(f"[aux] epoch {epoch + 1}/{args.aux_epochs} train {tot:.2f} val {vloss:.4f}", flush=True)
        if vloss < best:
            best = vloss
            torch.save(aux.state_dict(), os.path.join(out_dir, f"qwen_aux_layer{args.layer}_best.pt"))
    aux.load_state_dict(torch.load(os.path.join(out_dir, f"qwen_aux_layer{args.layer}_best.pt"),
                                   map_location=device))
    return aux.eval()


def train_sae(vf, tf, aux, args, device, out_dir):
    hidden = vf.shape[1] * args.hidden_ratio
    sae = VL_SAE(vf.shape[1], hidden, topk=args.topk).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.sae_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.sae_epochs)
    tr, va = split(len(vf)); va = va[:20480]  # cap val: full-val contrastive is O(n^2)
    best, patience = float("inf"), 0
    for epoch in range(args.sae_epochs):
        sae.train()
        perm = torch.randperm(len(tr))
        tot, steps = 0.0, 0
        for i in range(0, len(perm), args.sae_batch):
            b = perm[i:i + args.sae_batch]
            with torch.no_grad():
                be_v, be_t, _, _ = aux(vf[tr[b]].to(device), tf[tr[b]].to(device))
            opt.zero_grad()
            rv, rt, _, _ = sae(be_v, be_t)
            loss = nn.MSELoss()(rv, be_v) + nn.MSELoss()(rt, be_t)
            loss.backward()
            opt.step()
            tot += loss.item()
            steps += 1
        sched.step()
        sae.eval()
        with torch.no_grad():
            be_v, be_t, _, _ = aux(vf[va].to(device), tf[va].to(device))
            rv, rt, lv, lt = sae(be_v, be_t)
            vloss = (nn.MSELoss()(rv, be_v) + nn.MSELoss()(rt, be_t)).item()
            l0 = (lv > 0).float().sum(1).mean().item()
            dead = ((lv > 0).float().sum(0) == 0).float().mean().item()
        print(f"[sae] epoch {epoch + 1}/{args.sae_epochs} train {tot / max(steps, 1):.4f} "
              f"val {vloss:.4f} L0 {l0:.0f} dead {dead:.2%}", flush=True)
        if vloss < best:
            best, patience = vloss, 0
            torch.save(sae.state_dict(),
                       os.path.join(out_dir, f"qwen_vlsae_layer{args.layer}_{args.topk}_{args.hidden_ratio}_best.pth"))
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[sae] early stop at epoch {epoch + 1}", flush=True)
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--embeddings", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--hidden-ratio", type=int, default=8)
    ap.add_argument("--aux-epochs", type=int, default=30)
    ap.add_argument("--aux-batch", type=int, default=2048)
    ap.add_argument("--aux-lr", type=float, default=5e-5)
    ap.add_argument("--sae-epochs", type=int, default=30)
    ap.add_argument("--sae-batch", type=int, default=1024)
    ap.add_argument("--sae-lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--skip-aux", action="store_true", help="reuse existing aux checkpoint")
    ap.add_argument("--aux-only", action="store_true", help="train aux then exit (no SAE stage)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    emb = args.embeddings or os.path.join(here, "activations", f"qwen_cc3m_layer{args.layer}_merged.pt")
    out_dir = args.out_dir or os.path.join(here, "sae_weights")
    os.makedirs(out_dir, exist_ok=True)

    vf, tf = load_features(emb)
    print(f"[train] layer {args.layer}: {len(vf)} pairs, dim {vf.shape[1]}, "
          f"hidden {vf.shape[1] * args.hidden_ratio}, topk {args.topk}", flush=True)
    device = args.device
    if args.skip_aux:
        aux = AuxiliaryAE(vf.shape[1], tf.shape[1], projection_dim=vf.shape[1]).to(device)
        aux.load_state_dict(torch.load(os.path.join(out_dir, f"qwen_aux_layer{args.layer}_best.pt"),
                                       map_location=device))
        aux.eval()
        print(f"[train] reused aux checkpoint for layer {args.layer}", flush=True)
    else:
        aux = train_aux(vf, tf, args, device, out_dir)
    if args.aux_only:
        print(f"[train] DONE layer {args.layer} (aux only)", flush=True)
        return
    train_sae(vf, tf, aux, args, device, out_dir)
    print(f"[train] DONE layer {args.layer}", flush=True)


if __name__ == "__main__":
    main()
