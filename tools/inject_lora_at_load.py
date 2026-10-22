"""Inject HyperFlow LoRA deltas into TRT-LLM MiniMax-H3 weights at load time.

Two modes:
  install_load_time_lora(ModelClass, lora_path)   # add deltas during load_weights (no disk)
  fuse_to_disk(src_dir, dst_dir, lora_path)       # bake a fused checkpoint once (fast reloads)

Deltas are cached in bf16 so repeated runs skip the B@A recompute and avoid fp32
temporaries (the first version materialised fp32 copies of every target weight,
which made loading ~2x slower than it needed to be).
"""
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SKIP = ("time_embedder", "endpoint_time_embedder")
CACHE = "/work/models_hub/hyperflow/hf8_deltas_bf16.safetensors"


def deltas(lora_path, cache=CACHE):
    """Return {base_weight_key: delta(bf16)} with deltas cached on disk."""
    if cache and os.path.exists(cache):
        return {k: v for k, v in load_file(cache).items()}

    out = {}
    with safe_open(lora_path, framework="pt") as f:
        md = f.metadata() or {}
        rank = float(md.get("lora_rank", 256))
        scale = float(md.get("lora_alpha", 256)) / rank
        keys = list(f.keys())
        for k in keys:
            if not k.endswith(".lora_A.weight"):
                continue
            mod = k[: -len(".lora_A.weight")]
            if any(s in mod for s in SKIP):
                continue
            bk = mod + ".lora_B.weight"
            if bk not in keys:
                continue
            a = f.get_tensor(k).to(torch.float32)
            b = f.get_tensor(bk).to(torch.float32)
            out[mod.replace("transformer.", "", 1) + ".weight"] = ((b @ a) * scale).to(torch.bfloat16)

    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        save_file(out, cache)
        print(f"[lora-delta] cached {len(out)} deltas -> {cache}", flush=True)
    return out


def install_load_time_lora(model_cls, lora_path):
    d = deltas(lora_path)
    orig, state = model_cls.load_weights, {"count": 0}

    def load_weights(self, weights):
        hit = 0
        for k, dv in d.items():
            w = weights.get(k)
            if w is None:
                continue
            # bf16 add on the target device: no fp32 materialisation
            weights[k] = w + dv.to(device=w.device, dtype=w.dtype)
            hit += 1
        state["count"] += 1
        print(f"[lora-inject] pass {state['count']}: {hit}/{len(d)} keys hit", flush=True)
        return orig(self, weights)

    model_cls.load_weights = load_weights
    return d


def fuse_to_disk(src_dir, dst_dir, lora_path):
    """Write a checkpoint with deltas baked in. Engines then load plain weights."""
    d = deltas(lora_path)
    os.makedirs(dst_dir, exist_ok=True)
    idx = json.load(open(os.path.join(src_dir, "diffusion_pytorch_model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    hit = 0
    for shard in shards:
        w = load_file(os.path.join(src_dir, shard))
        for k in list(w.keys()):
            dv = d.get(k)
            if dv is not None:
                w[k] = w[k] + dv.to(device=w[k].device, dtype=w[k].dtype)
                hit += 1
        save_file(w, os.path.join(dst_dir, shard))
        print(f"[fuse] {shard} done (hits so far {hit})", flush=True)
    for extra in ("config.json",):
        s = os.path.join(src_dir, extra)
        if os.path.exists(s):
            open(os.path.join(dst_dir, extra), "wb").write(open(s, "rb").read())
    open(os.path.join(dst_dir, "diffusion_pytorch_model.safetensors.index.json"), "w").write(json.dumps(idx))
    print(f"[fuse] fused {hit}/{len(d)} keys -> {dst_dir}", flush=True)
    return hit
