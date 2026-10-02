# VL-SAE for Qwen2.5-VL

Extension of VL-SAE to **Qwen2.5-VL-3B-Instruct** (decoder layer 26):
activation collection and two-stage VL-SAE training (AuxiliaryAE + TopK SAE).

The model classes (`AuxiliaryAE`, `VL_SAE`) are the upstream ones from
`lvlms/sae_trainer/sae_model.py` — nothing is redefined here.

Downstream concept analysis (grounding audit, paper-style CLIP intra/inter
concept evaluation, concept visualization) works unchanged with the repo's
existing tooling under `lvlms/eval` and `lvlms/demo` — point it at the
activations and weights produced here.

## Layout

| file | stage |
|---|---|
| `qwen_collect_activations.py` | layer-26 activation collection over image-text pairs (one forward per pair, with image) |
| `merge_shards.py` | merge per-shard activation dumps into one `.pt` |
| `train_qwen_sae.py` | two-stage training: aux (InfoNCE τ=0.07 + recon MSE) → SAE (per-modality MSE); `--aux-only`, `--skip-aux` |

## Configuration (environment variables)

| var | default | meaning |
|---|---|---|
| `QWEN_VL_MODEL_PATH` | `Qwen/Qwen2.5-VL-3B-Instruct` | model checkpoint (local path or HF id) |
| `VLSAE_QWEN_WORKDIR` | `<this dir>/qwen_workdir` | data/weights/results root (activations, trained weights) |

The workdir layout produced/consumed by the scripts:

```
qwen_workdir/
  activations/qwen_cc3m_layer26_merged.pt
  sae_weights/qwen_aux_layer26_best.pt
  sae_weights/qwen_vlsae_layer26_128_8_best.pth
```

## Quick start

```bash
# 1. collect activations (pairs jsonl: {"image": rel_path, "text": caption})
python qwen_collect_activations.py --shard 0 --num-shards 8 --batch-size 16 \
    --pairs all_pairs.jsonl --image-root $VLSAE_QWEN_WORKDIR \
    --out-dir $VLSAE_QWEN_WORKDIR/activations
python merge_shards.py --act-dir $VLSAE_QWEN_WORKDIR/activations

# 2. train (aux 50ep per paper recipe, then SAE)
python train_qwen_sae.py --layer 26 --aux-epochs 50 --aux-only \
    --embeddings $VLSAE_QWEN_WORKDIR/activations/qwen_cc3m_layer26_merged.pt \
    --out-dir $VLSAE_QWEN_WORKDIR/sae_weights
python train_qwen_sae.py --layer 26 --skip-aux --sae-epochs 30 \
    --embeddings $VLSAE_QWEN_WORKDIR/activations/qwen_cc3m_layer26_merged.pt \
    --out-dir $VLSAE_QWEN_WORKDIR/sae_weights
```

Note: the aux `projection_dim` is set to the feature dim (2048 here) — the
upstream default of 4096 matches the LLaVA hidden size, not Qwen2.5-VL-3B.

## Notes from our runs (2.9M CC3M pairs, Qwen2.5-VL-3B, layer 26)

- **Aux training budget matters.** With a short aux schedule the cosine
  encoder collapses (near-zero semantically evaluable concepts); with the
  full 50-epoch paper recipe it becomes semantically evaluable. A large
  share of the cosine encoder's apparent weakness is an aux-undertraining
  artifact, not an architecture flaw.
