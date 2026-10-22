# Running HyperFlow 8-step on engines without a `TwoTimeEmbedder` hook

Status: **AdaLN folding verified numerically** (table computed, 168× smaller than the MLP it replaces).
End-to-end PSNR validation is the next step.

## Why fold

HyperFlow is *not* a generic LoRA. Its adapter carries `endpoint_time_embedder.*` keys that only
exist once `TwoTimeEmbedder` is installed, so it can only be loaded through
`load_hyperflow_lora` on the diffusers Modular Pipeline. Engines with a fixed model definition
(TRT-LLM, vLLM-Omni, SGLang) have nowhere to put those two modules.

But `TwoTimeEmbedder.forward` (`src/hyperflow_h3/embedder.py`) is a pure function of the two
timestep scalars and nothing else:

```python
def forward(self, sample):                      # `sample` is a timestep tensor
    t_emb = self.base(sample)                   # LoRA-corrected time_embedder
    r_proj = self.time_proj(endpoint)           # endpoint = r
    r_emb = self.endpoint(r_proj)               # LoRA-corrected endpoint_time_embedder
    return t_emb + self.gate * (r_emb - t_emb)   # = (1-g)·base(t) + g·endpoint(r)
```

With a fixed 8-step grid there are only a handful of `(t, r)` pairs, so the whole thing — the two
embedders, their LoRA deltas, the gate, **and** the per-block `adaln_proj` MLP downstream of it —
collapses into a lookup table indexed by `(step, modality)`.

## Fold formula

```python
t_i = 1 - shift(sigma_i,   sh)
r_i = 1 - shift(sigma_{i+1}, sh)     # note r_i == t_{i+1}
temb = (1 - gate) * base(t_i) + gate * endpoint(r_i)
table[block, row] = adaln_proj_block(silu(temb))
```

with, straight out of the checkpoint header:

| key | value |
|---|---|
| `hyperflow_gate` | 0.25 |
| `hyperflow_sigmas` | `[1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0]` |
| `hyperflow_video_shift` | 12.0 |
| `hyperflow_audio_shift` | 3.0 |
| `lora_rank` / `lora_alpha` | 256 / 256 (scale 1.0) |
| `compatible_subfolders` | `["transformer", "transformer_ref"]` |

Rows: `step * 2 + modality`, `video = 0`, `audio = 1` → 16 rows.

## Verified numbers (MiniMax-H3, GB300, `Atten`-free CPU fold)

```
time_embedder           256 -> 5376 -> 2688   (LoRA-corrected)
endpoint_time_embedder  256 -> 5376 -> 2688   (LoRA-only, no base weights)
lora modules            316  (includes both time embedders; adaln_proj is NOT a target)
adaln_proj              50 separate per-block modules, out_dim = 96768 = 6 * 5376 * 3
table                   50 blocks x 16 rows x 96768  ->  154.8 MB (bf16)
replaced                26 GB of adaln_proj + all per-step embedder compute
compression             ~168x
```

Self-check that passed: `r_i == t_{i+1}` for consecutive rows.

## What still needs a code change

Only the `adaln_proj` call site. Per block:

```python
if self.use_adaln_table:
    mod = F.embedding(row_idx, self.adaln_table)   # row_idx = step * 2 + modality
else:
    mod = self.adaln_proj.table(temb)              # original path
```

Everything else is native:

| piece | mechanism |
|---|---|
| attention / FFN LoRA (308 modules, rank 256) | engine runtime LoRA — no fusion |
| time embedders + their LoRA + gate | folded into the table |
| `adaln_proj` (50 x 26 GB total) | replaced by the table |
| sampling schedule | the 8-step sigma grid above, fixed |

**Quantization caveat:** TRT-LLM's runtime LoRA cannot target quantized components, so FP8/NVFP4
runs need the block LoRA fused into the weights first.

## Reproduce

```bash
python tools/fold_adaln_table.py \
  --hyperflow  /path/to/minimax_h3_hyperflow_8step_v1.0.safetensors \
  --base-index /path/to/MiniMax-H3/transformer/diffusion_pytorch_model.safetensors.index.json \
  --out        /path/to/hf8_adaln_table.safetensors
```
