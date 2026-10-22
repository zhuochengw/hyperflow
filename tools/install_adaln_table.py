"""Install folded AdaLN tables into a TRT-LLM MiniMax-H3 transformer at runtime.

The folded table produced by `tools/fold_adaln_table.py` replaces both HyperFlow's
`TwoTimeEmbedder` and the per-block `adaln_proj` MLP. Serving it needs one hook: map
`(step, modality)` to a table row instead of running the MLP.

This is done by monkeypatching `table()` on each block's
`MiniMaxH3AdaLayerNormModulation`. Both TRT-LLM call sites go through `.table()`, so one patch
covers `self.adaln_proj(temb)` and the fused-adaLN path alike. No source edit, so it survives
TRT-LLM upgrades and gives a clear failure mode if the module layout changes.

Usage:
    from safetensors.torch import load_file
    from tools.install_adaln_table import install_adaln_table, set_row

    table = load_file("hf8_adaln_table.safetensors")
    install_adaln_table(transformer, table, num_modalities=2)
    for step in range(8):
        set_row(transformer, step * 2 + 0)      # video
        # ... forward ...
        set_row(transformer, step * 2 + 1)      # audio
"""
import types

import torch


def _blocks_with_adaln(transformer):
    for name, module in transformer.named_modules():
        if hasattr(module, "adaln_proj") and hasattr(module.adaln_proj, "table"):
            yield name, module


def install_adaln_table(transformer, table_by_block, num_modalities=2, verbose=True):
    """Replace each block's adaln_proj MLP with a lookup into `table_by_block`.

    Args:
        transformer: the MiniMax-H3 transformer module.
        table_by_block: mapping ``"blocks.{i}.adaln_table" -> Tensor[rows, 6*hidden]``.
        num_modalities: rows per step (video, audio) = 2.

    Returns:
        number of blocks patched.
    """
    patched = 0
    for idx, (name, module) in enumerate(_blocks_with_adaln(transformer)):
        key = f"blocks.{idx}.adaln_table"
        if key not in table_by_block:
            raise KeyError(f"folded table missing {key} (block {name})")
        table = table_by_block[key]

        def make_table_override(tbl, blk):
            def table(self, temb):  # noqa: ARG001 - signature must match the original
                row = getattr(self, "_adaln_row", 0)
                return tbl[row: row + 1].to(device=temb.device, dtype=temb.dtype)
            return table

        module.adaln_proj.table = types.MethodType(
            make_table_override(table, module), module.adaln_proj
        )
        module.adaln_proj._adaln_table_rows = int(table.shape[0])
        patched += 1

    if verbose:
        print(f"[adaln-table] patched {patched} blocks, {num_modalities} modalities/step")
    return patched


def set_row(transformer, row):
    """Select the folded `(step, modality)` row on every patched block."""
    for _, module in _blocks_with_adaln(transformer):
        module.adaln_proj._adaln_row = int(row)
