import json
import torch
from safetensors import safe_open

def deltas(path, skip=("time_embedder", "endpoint_time_embedder")):
    out = {}
    with safe_open(path, framework="pt") as f:
        md = f.metadata() or {}
        rank = float(md.get("lora_rank", 256))
        scale = float(md.get("lora_alpha", 256)) / rank
        ks = list(f.keys())
        for k in ks:
            if not k.endswith(".lora_A.weight"):
                continue
            mod = k[: -len(".lora_A.weight")]
            if any(s in mod for s in skip):
                continue
            bk = mod + ".lora_B.weight"
            if bk not in ks:
                continue
            a = f.get_tensor(k).float()
            b = f.get_tensor(bk).float()
            key = mod.replace("transformer.", "", 1) + ".weight"
            out[key] = (b @ a) * scale
    return out

def install(model_cls, lora_path):
    d = deltas(lora_path)
    orig = model_cls.load_weights

    def load_weights(self, weights):
        hit = 0
        for k, dv in d.items():
            if k in weights:
                ref = weights[k]
                weights[k] = (ref.float() + dv.to(ref.device)).to(ref.dtype)
                hit += 1
        print(f"[lora-inject] {hit}/{len(d)} keys hit", flush=True)
        if hit == 0:
            print("[lora-inject] WARNING: no keys matched", flush=True)
        return orig(self, weights)

    model_cls.load_weights = load_weights
    return d
