"""Fold HyperFlow's two-time conditioning + per-block AdaLN MLP into a lookup table.

HyperFlow's `TwoTimeEmbedder` is a pure function of the (t, r) timestep pair, and with a fixed
8-step grid there are only 16 distinct (step, modality) rows. So the two time embedders (with
their rank-256 LoRA), the gate blend, and the 50 per-block `adaln_proj` MLPs all collapse into
one table indexed by `step * 2 + modality`.

Usage:
    python tools/fold_adaln_table.py \
        --hyperflow  minimax_h3_hyperflow_8step_v1.0.safetensors \
        --base-index MiniMax-H3/transformer/diffusion_pytorch_model.safetensors.index.json \
        --out        hf8_adaln_table.safetensors
"""
import argparse
import json
import math
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

VIDEO, AUDIO = 0, 1


def shift_sigma(s, sh):
    """Flow-matching sigma shift used by HyperFlow (`hyperflow_{video,audio}_shift`)."""
    return sh * s / (1.0 + (sh - 1.0) * s)


def timestep_embedding(t, dim=256, flip_sin_to_cos=True, downscale_freq_shift=1.0):
    """diffusers `Timesteps` for MiniMax-H3."""
    half = dim // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32) / (half - downscale_freq_shift)
    emb = t.float().reshape(-1, 1) * torch.exp(exponent).reshape(1, -1)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
    return emb


class BaseReader:
    """Random access into the sharded diffusers checkpoint via its index."""

    def __init__(self, index_path):
        self.dir = os.path.dirname(index_path)
        self.weight_map = json.load(open(index_path))["weight_map"]
        self._open = {}

    def get(self, key, dtype=torch.float32):
        shard = self.weight_map[key]
        if shard not in self._open:
            self._open[shard] = safe_open(os.path.join(self.dir, shard), framework="pt")
        return self._open[shard].get_tensor(key).to(dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyperflow", required=True, help="HyperFlow .safetensors")
    ap.add_argument("--base-index", required=True, help="MiniMax-H3 transformer index json")
    ap.add_argument("--out", required=True, help="output table .safetensors")
    ap.add_argument("--modalities", default="video,audio")
    args = ap.parse_args()

    meta = {}
    with safe_open(args.hyperflow, framework="pt") as f:
        meta = f.metadata() or {}
        hf = {k: f.get_tensor(k).float() for k in f.keys()}

    gate = float(meta["hyperflow_gate"])
    sigmas = json.loads(meta["hyperflow_sigmas"])
    shifts = {
        "video": float(meta["hyperflow_video_shift"]),
        "audio": float(meta["hyperflow_audio_shift"]),
    }
    scale = float(meta["lora_alpha"]) / float(meta["lora_rank"])

    base = BaseReader(args.base_index)

    def lora_delta(module):
        a = hf[f"transformer.{module}.lora_A.weight"]
        b = hf[f"transformer.{module}.lora_B.weight"]
        return (b @ a) * scale

    def blended_embedder(prefix):
        w1 = base.get(f"{prefix}.linear_1.weight") + lora_delta(f"{prefix}.linear_1")
        w2 = base.get(f"{prefix}.linear_2.weight") + lora_delta(f"{prefix}.linear_2")
        return w1, w2

    # endpoint embedder has no base weights in MiniMax-H3: its weights are pure LoRA
    w1, w2 = blended_embedder("time_embedder")
    e1 = lora_delta("endpoint_time_embedder.linear_1")
    e2 = lora_delta("endpoint_time_embedder.linear_2")

    def temb(t):
        h = torch.nn.functional.silu(timestep_embedding(t) @ w1.T) @ w2.T
        e = torch.nn.functional.silu(timestep_embedding(t) @ e1.T) @ e2.T
        return (1.0 - gate) * h + gate * e

    modalities = args.modalities.split(",")
    rows, labels = [], []
    for m in modalities:
        sh = shifts[m]
        for i in range(len(sigmas) - 1):
            t = 1.0 - shift_sigma(sigmas[i], sh)
            r = 1.0 - shift_sigma(sigmas[i + 1], sh)
            rows.append(temb(torch.tensor([t]))[0])
            labels.append({"step": i, "modality": m, "t": float(t), "r": float(r)})

    temb_all = torch.stack(rows)
    act = torch.nn.functional.silu(temb_all)

    blocks = sorted({int(k.split(".")[1]) for k in base.weight_map if "adaln_proj.linear.weight" in k})
    table = {}
    for b in blocks:
        w = base.get(f"transformer_blocks.{b}.adaln_proj.linear.weight")
        bias = base.get(f"transformer_blocks.{b}.adaln_proj.linear.bias")
        table[f"blocks.{b}.adaln_table"] = (act @ w.T + bias).to(torch.bfloat16).contiguous()

    save_file(table, args.out)
    meta_out = os.path.splitext(args.out)[0] + ".json"
    json.dump(
        {
            "gate": gate,
            "sigmas": sigmas,
            "shifts": shifts,
            "row_index": "step * n_modalities + modality_index",
            "modalities": modalities,
            "rows": labels,
            "num_blocks": len(blocks),
            "out_dim": int(act.shape[-1]),
        },
        open(meta_out, "w"),
        indent=2,
    )
    print(f"blocks={len(blocks)} rows={act.shape[0]} out_dim={act.shape[-1]}")
    print(f"saved {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB) and {meta_out}")


if __name__ == "__main__":
    main()
