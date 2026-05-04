"""
Reproducer + fix verification for the hidden_layer_index bug observed in
Day 0 M1 vs M2.

The bug: M1 (L=-1) and M2 (L=-18) produced byte-identical val curves and
SLPA `V_mean` because the actor worker is constructed with
`config=actor_rollout_ref` (no `algorithm` field), so the fallback that
populates `meta_info["hidden_layer_index"]` in `fsdp_workers.py` was dead
code (`hasattr(self.config, "algorithm")` → False). dp_actor.py then read
the meta_info with default -1, regardless of slpa/scr.repr.hidden_layer_index.

The fix is in ray_trainer.py: set `batch.meta_info["hidden_layer_index"]`
on the trainer side, where `self.config.algorithm` IS available, right
where we set the other meta flags before `compute_log_prob`.

Two stages:
  Stage 1 (CPU, ~1s): pure-PyTorch ModuleList sanity check that hook
    semantics work — confirms the fix lives upstream of dp_actor, not in
    the hook code.
  Stage 2 (CPU, ~5s): construct fake configs that mimic actor-worker
    vs trainer-side config layouts, walk the fix path, assert
    `meta_info["hidden_layer_index"]` ends up at the right value.
"""
from __future__ import annotations
import sys
import torch
from torch import nn


def stage1_hook_semantics():
    """Confirm: a hook on ModuleList[-18] does capture layer 18 output."""
    print("=== Stage 1: PyTorch hook semantics on a fake 36-layer ModuleList ===")

    class TaggedLayer(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.idx = idx
            self.scale = nn.Parameter(torch.tensor(float(idx + 1)), requires_grad=False)

        def forward(self, x):
            return x + self.scale

    class ToyModel(nn.Module):
        def __init__(self, n_layers=36):
            super().__init__()
            self.layers = nn.ModuleList([TaggedLayer(i) for i in range(n_layers)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = ToyModel(36)
    blocks = model.layers
    assert len(blocks) == 36

    captured = {"out": None}
    def _hook(_m, _i, output):
        captured["out"] = output

    target = blocks[-18]
    handle = target.register_forward_hook(_hook)
    final = model(torch.tensor(0.0))
    handle.remove()

    expected_capture = sum(i + 1 for i in range(19))  # layers 0..18 → 190
    expected_final = sum(i + 1 for i in range(36))  # layers 0..35 → 666

    cap = captured["out"].item()
    fin = final.item()
    print(f"  hook captured (layer 18) = {cap} (expect {expected_capture})")
    print(f"  final output (layer 35)  = {fin} (expect {expected_final})")

    ok = (cap == expected_capture) and (fin == expected_final)
    print(f"  {'PASS' if ok else 'FAIL'} — hook semantics are fine; bug is NOT here\n")
    return ok


def stage2_meta_info_propagation():
    """Verify the trainer-side fix: `batch.meta_info['hidden_layer_index']`
    gets populated from the first enabled module's repr config.
    """
    print("=== Stage 2: meta_info propagation logic (replicates ray_trainer fix) ===")

    class FakeRepr:
        def __init__(self, idx, pooling="last"):
            self.hidden_layer_index = idx
            self.pooling = pooling

    class FakeCfg:
        def __init__(self, enable, idx):
            self.enable = enable
            self.repr = FakeRepr(idx)

    def populate(meta_info, sentence_adv_cfg, slpa_cfg, scr_cfg):
        """Lifted from the ray_trainer.py fix."""
        for _cfg in (sentence_adv_cfg, slpa_cfg, scr_cfg):
            if _cfg is not None and bool(getattr(_cfg, "enable", False)):
                _repr_cfg = getattr(_cfg, "repr", None)
                if _repr_cfg is not None:
                    _layer = getattr(_repr_cfg, "hidden_layer_index", -1)
                    if isinstance(_layer, int):
                        meta_info["hidden_layer_index"] = _layer
                    else:
                        try:
                            meta_info["hidden_layer_index"] = [int(i) for i in _layer]
                        except TypeError:
                            meta_info["hidden_layer_index"] = -1
                    break
        return meta_info

    cases = [
        # (description, sent_adv, slpa, scr, expected)
        ("M1: slpa enabled, L=-1", None, FakeCfg(True, -1), FakeCfg(True, -1), -1),
        ("M2: slpa enabled, L=-18", None, FakeCfg(True, -18), FakeCfg(True, -18), -18),
        ("scr only, L=-9", None, FakeCfg(False, -1), FakeCfg(True, -9), -9),
        ("multi-layer ensemble [-1,-9,-18]", None, FakeCfg(True, [-1, -9, -18]), FakeCfg(True, [-1, -9, -18]), [-1, -9, -18]),
        ("sentence_adv overrides slpa", FakeCfg(True, -27), FakeCfg(True, -18), FakeCfg(True, -18), -27),
    ]

    all_ok = True
    for desc, sa, sl, sc, expected in cases:
        meta = {}
        populate(meta, sa, sl, sc)
        got = meta.get("hidden_layer_index")
        ok = got == expected
        all_ok &= ok
        mark = "✓" if ok else "✗"
        print(f"  {mark} {desc}: got {got}, expected {expected}")

    print(f"  {'PASS' if all_ok else 'FAIL'} — meta_info propagation logic is correct\n")
    return all_ok


def main():
    ok1 = stage1_hook_semantics()
    ok2 = stage2_meta_info_propagation()
    print("=== summary ===")
    print(f"  stage1 (hook semantics): {'PASS' if ok1 else 'FAIL'}")
    print(f"  stage2 (meta_info fix):  {'PASS' if ok2 else 'FAIL'}")
    if ok1 and ok2:
        print("\n  Fix logic verified. Next: run a fresh M2 (L=-18) experiment;")
        print("  it should produce slpa/V_mean differing from M1 (L=-1) at every step.")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
