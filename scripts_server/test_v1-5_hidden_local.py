#!/usr/bin/env python3
"""Local sanity checks for v1-5-hidden code changes.

Runs on CPU. No ray/vllm/HF needed. Exercises:
  1. pool_sentence_embeddings — all 6 pooling strategies, shapes, sample-idx.
  2. Backward compat — "last" pooling matches the pre-change implementation.
  3. _select_hidden_layer — int / list / 1-elem / empty / ListConfig-like / bool.
  4. _compute_repr_diagnostic_metrics — keys, no NaNs, edge cases.
  5. SentenceReprConfig defaults + repr field on each adv config dataclass.
  6. build_punct_token_ids on a mock tokenizer.

Usage:
    cd /root/sentencepo_v1-5
    python3 scripts_server/test_v1-5_hidden_local.py
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------- helpers ----------------------------

PASS, FAIL = 0, 0
FAILURES: list[tuple[str, str]] = []


def case(name: str):
    def deco(fn):
        def runner():
            global PASS, FAIL
            try:
                fn()
                PASS += 1
                print(f"[PASS] {name}")
            except Exception:  # noqa: BLE001
                FAIL += 1
                tb = traceback.format_exc()
                FAILURES.append((name, tb))
                print(f"[FAIL] {name}")
                print(tb)
        runner.__name__ = fn.__name__
        return runner
    return deco


def assert_close(a, b, rtol=1e-5, atol=1e-6, msg=""):
    a = a if isinstance(a, torch.Tensor) else torch.tensor(a)
    b = b if isinstance(b, torch.Tensor) else torch.tensor(b)
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg} (max abs diff {diff:.3e})\n  a={a}\n  b={b}")


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load sentence_repr directly (zero verl deps).
sr = load_module("sentence_repr", str(PROJECT_ROOT / "verl/utils/sentence_repr.py"))

# Extract _select_hidden_layer from dp_actor.py without importing the whole module.
import ast

_dp_src = (PROJECT_ROOT / "verl/workers/actor/dp_actor.py").read_text()
_tree = ast.parse(_dp_src)
_helper_src = None
for _node in _tree.body:
    if isinstance(_node, ast.FunctionDef) and _node.name == "_select_hidden_layer":
        _helper_src = ast.get_source_segment(_dp_src, _node)
        break
assert _helper_src is not None, "could not find _select_hidden_layer"
_ns = {"torch": torch}
exec(_helper_src, _ns)
_select_hidden_layer = _ns["_select_hidden_layer"]

# Extract _compute_repr_diagnostic_metrics similarly.
_ca_src = (PROJECT_ROOT / "verl/trainer/ppo/core_algos.py").read_text()
_ca_tree = ast.parse(_ca_src)
_diag_src = None
for _node in _ca_tree.body:
    if isinstance(_node, ast.FunctionDef) and _node.name == "_compute_repr_diagnostic_metrics":
        _diag_src = ast.get_source_segment(_ca_src, _node)
        break
assert _diag_src is not None, "could not find _compute_repr_diagnostic_metrics"
exec(_diag_src, _ns)
_compute_repr_diagnostic_metrics = _ns["_compute_repr_diagnostic_metrics"]


# ---------------------------- fixtures ----------------------------

def build_fixture(seed: int = 0, hidden: int = 8, bs: int = 3, seqlen: int = 12):
    """Synthetic batch with 3 samples × 12 tokens, varied sentence layouts.

    sentence_ids use the production convention: globally unique across samples
    (e.g. ray_trainer.py adds sample_idx * (seqlen + 1) per sample).
    """
    torch.manual_seed(seed)
    h = torch.randn(bs, seqlen, hidden)

    # response_mask: first 4 tokens are prompt (mask 0), then 8 are response (mask 1).
    response_mask = torch.zeros(bs, seqlen, dtype=torch.long)
    response_mask[:, 4:] = 1

    sid = torch.full((bs, seqlen), -1, dtype=torch.long)
    # sample 0: 1 sentence (id 0+offset), 8 tokens
    sid[0, 4:] = 0 + 0 * (seqlen + 1)
    # sample 1: 2 sentences (4+4 tokens)
    sid[1, 4:8] = 0 + 1 * (seqlen + 1)
    sid[1, 8:] = 1 + 1 * (seqlen + 1)
    # sample 2: 3 sentences (3+3+2 tokens)
    sid[2, 4:7] = 0 + 2 * (seqlen + 1)
    sid[2, 7:10] = 1 + 2 * (seqlen + 1)
    sid[2, 10:] = 2 + 2 * (seqlen + 1)
    return h, sid, response_mask


# ---------------------------- tests ----------------------------

@case("pool_sentence_embeddings: shapes and sample_idx")
def t_shapes():
    h, sid, mask = build_fixture()
    out = sr.pool_sentence_embeddings(h, sid, mask, "last")
    assert out is not None
    sent_emb, uid, sample_idx = out
    # Total sentences: 1 + 2 + 3 = 6.
    assert sent_emb.shape == (6, 8), sent_emb.shape
    assert uid.shape == (6,)
    assert sample_idx.shape == (6,)
    assert sample_idx.tolist() == sorted(sample_idx.tolist())
    # Verify sample_idx assignment.
    counts = torch.bincount(sample_idx, minlength=3).tolist()
    assert counts == [1, 2, 3], counts


@case("pool_sentence_embeddings: 'last' matches old implementation byte-for-byte")
def t_backcompat_last():
    """Reproduce the OLD last-token-only pooling and compare."""
    h, sid, mask = build_fixture()
    valid = (mask > 0) & (sid >= 0)
    flat_sid = sid.view(-1)
    flat_valid = valid.view(-1)
    flat_sid_valid = flat_sid[flat_valid]
    unique_sid, _ = torch.unique(flat_sid_valid, return_inverse=True)
    next_sid = torch.roll(sid, shifts=-1, dims=1)
    next_valid = torch.roll(valid, shifts=-1, dims=1)
    last_pos_mask = torch.zeros_like(valid)
    last_pos_mask[:, -1] = True
    boundary = last_pos_mask | (sid != next_sid) | (~next_valid)
    last_mask = valid & boundary
    flat_last = last_mask.view(-1)
    idx_last = torch.searchsorted(unique_sid, flat_sid[flat_last])
    flat_emb_last = h.view(-1, h.shape[-1])[flat_last]
    expected = torch.zeros((unique_sid.numel(), h.shape[-1]))
    expected.index_copy_(0, idx_last, flat_emb_last)

    new_emb, _, _ = sr.pool_sentence_embeddings(h, sid, mask, "last")
    assert_close(new_emb, expected, msg="last pooling regressed vs old implementation")


@case("pool_sentence_embeddings: 'first' picks first valid token")
def t_first():
    h, sid, mask = build_fixture()
    out = sr.pool_sentence_embeddings(h, sid, mask, "first")
    assert out is not None
    sent_emb, uid, _ = out
    # sample 1 sentence 1 starts at position 8 → emb should equal h[1, 8]
    # uid for sentence at sample 1 index 1 = 1 + 1*13 = 14
    target_uid = 1 + 1 * 13
    where = (uid == target_uid).nonzero(as_tuple=True)[0]
    assert where.numel() == 1
    assert_close(sent_emb[where[0]], h[1, 8], msg="first pooling wrong position")


@case("pool_sentence_embeddings: 'mean' equals mean over response tokens of sentence")
def t_mean():
    h, sid, mask = build_fixture()
    sent_emb, uid, _ = sr.pool_sentence_embeddings(h, sid, mask, "mean")
    # sample 0 sentence 0 covers tokens 4..12 of sample 0, uid = 0
    target_uid = 0
    idx = (uid == target_uid).nonzero(as_tuple=True)[0]
    assert idx.numel() == 1
    expected_mean = h[0, 4:].mean(dim=0)
    assert_close(sent_emb[idx[0]], expected_mean, msg="mean pooling mismatch")


@case("pool_sentence_embeddings: 'diff' equals last - first")
def t_diff():
    h, sid, mask = build_fixture()
    last_emb, uid_l, _ = sr.pool_sentence_embeddings(h, sid, mask, "last")
    first_emb, uid_f, _ = sr.pool_sentence_embeddings(h, sid, mask, "first")
    diff_emb, uid_d, _ = sr.pool_sentence_embeddings(h, sid, mask, "diff")
    assert torch.equal(uid_l, uid_d) and torch.equal(uid_f, uid_d)
    assert_close(diff_emb, last_emb - first_emb, msg="diff pooling != last - first")


@case("pool_sentence_embeddings: 'mean_no_punct' falls back to last when sentence is all-punct")
def t_mean_no_punct_fallback():
    h, sid, mask = build_fixture()
    # Build response_token_ids: every token id = 99 EXCEPT we'll mark token id 1 as punct.
    response_token_ids = torch.full_like(sid, 99)
    # Make sample 0's entire sentence consist only of "punct" token id (1).
    response_token_ids[0, 4:] = 1
    punct_token_ids = torch.tensor([1], dtype=torch.long)
    sent_emb, uid, _ = sr.pool_sentence_embeddings(
        h, sid, mask, "mean_no_punct",
        response_token_ids=response_token_ids,
        punct_token_ids=punct_token_ids,
    )
    # Sample 0 sentence (uid 0): all punct → fallback to last → equals h[0, -1]
    idx0 = (uid == 0).nonzero(as_tuple=True)[0][0]
    assert_close(sent_emb[idx0], h[0, -1], msg="mean_no_punct fallback failed")


@case("pool_sentence_embeddings: 'mean_no_punct' excludes punct tokens")
def t_mean_no_punct_normal():
    h, sid, mask = build_fixture()
    response_token_ids = torch.full_like(sid, 99)
    # Sample 1 sentence 0 = positions 4..8, mark position 4 as punct.
    response_token_ids[1, 4] = 1
    punct_token_ids = torch.tensor([1], dtype=torch.long)
    sent_emb, uid, _ = sr.pool_sentence_embeddings(
        h, sid, mask, "mean_no_punct",
        response_token_ids=response_token_ids,
        punct_token_ids=punct_token_ids,
    )
    # uid for sample 1 sentence 0: 0 + 1*13 = 13
    idx = (uid == 13).nonzero(as_tuple=True)[0][0]
    expected = h[1, 5:8].mean(dim=0)  # positions 5,6,7 (4 is punct)
    assert_close(sent_emb[idx], expected, msg="mean_no_punct should drop punct tokens")


@case("pool_sentence_embeddings: 'entropy_weighted' down-weights high entropy")
def t_entropy_weighted():
    h, sid, mask = build_fixture()
    # Set all entropies the same → equivalent to plain mean.
    ent = torch.zeros_like(mask, dtype=torch.float32)
    sent_emb_eq, _, _ = sr.pool_sentence_embeddings(
        h, sid, mask, "entropy_weighted", token_entropy=ent
    )
    sent_emb_mean, _, _ = sr.pool_sentence_embeddings(h, sid, mask, "mean")
    assert_close(sent_emb_eq, sent_emb_mean, msg="entropy_weighted with const entropy != mean")
    # Now set a single token's entropy very high → shifts result.
    ent[1, 4] = 1e6  # weight ~ 0
    sent_emb_skewed, uid, _ = sr.pool_sentence_embeddings(
        h, sid, mask, "entropy_weighted", token_entropy=ent
    )
    # uid 13 = sample 1 sent 0; expected ≈ mean of h[1, 5:8]
    idx = (uid == 13).nonzero(as_tuple=True)[0][0]
    expected = h[1, 5:8].mean(dim=0)
    assert torch.allclose(sent_emb_skewed[idx], expected, atol=1e-3), (
        f"entropy_weighted didn't down-weight high-entropy token enough: "
        f"got {sent_emb_skewed[idx]}, expected {expected}"
    )


@case("pool_sentence_embeddings: invalid pooling raises")
def t_invalid_pool():
    h, sid, mask = build_fixture()
    try:
        sr.pool_sentence_embeddings(h, sid, mask, "bogus")
    except ValueError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("expected ValueError")


@case("pool_sentence_embeddings: empty (no valid tokens) returns None")
def t_empty():
    h = torch.randn(2, 4, 8)
    sid = torch.full((2, 4), -1, dtype=torch.long)
    mask = torch.zeros(2, 4, dtype=torch.long)
    assert sr.pool_sentence_embeddings(h, sid, mask, "last") is None


@case("_select_hidden_layer: int picks correct layer")
def t_select_int():
    layers = tuple(torch.full((1, 4, 3), float(i)) for i in range(5))
    out = _select_hidden_layer(layers, -1)
    assert_close(out, torch.full_like(out, 4.0))
    out = _select_hidden_layer(layers, 0)
    assert_close(out, torch.full_like(out, 0.0))
    out = _select_hidden_layer(layers, -3)
    assert_close(out, torch.full_like(out, 2.0))


@case("_select_hidden_layer: list mean-ensembles")
def t_select_list():
    layers = tuple(torch.full((1, 4, 3), float(i)) for i in range(5))
    out = _select_hidden_layer(layers, [-1, -3, -5])
    # mean of 4, 2, 0 = 2
    assert_close(out, torch.full_like(out, 2.0))


@case("_select_hidden_layer: 1-elem list = int")
def t_select_oneelem():
    layers = tuple(torch.full((1, 4, 3), float(i)) for i in range(5))
    out_int = _select_hidden_layer(layers, -2)
    out_list = _select_hidden_layer(layers, [-2])
    assert_close(out_int, out_list)


@case("_select_hidden_layer: empty list -> last")
def t_select_empty():
    layers = tuple(torch.full((1, 4, 3), float(i)) for i in range(5))
    out = _select_hidden_layer(layers, [])
    assert_close(out, layers[-1])


@case("_select_hidden_layer: ListConfig-like (custom non-list iterable) works")
def t_select_listlike():
    class FakeListConfig:
        """Mimics omegaconf.ListConfig: iterable, not subclass of list."""
        def __init__(self, items):
            self._items = items
        def __iter__(self):
            return iter(self._items)
        def __len__(self):
            return len(self._items)
    layers = tuple(torch.full((1, 4, 3), float(i)) for i in range(5))
    out = _select_hidden_layer(layers, FakeListConfig([-1, -3]))
    # mean of 4, 2 = 3
    assert_close(out, torch.full_like(out, 3.0))


@case("_select_hidden_layer: bool rejected (catches config typos)")
def t_select_bool():
    layers = tuple(torch.full((1, 4, 3), float(i)) for i in range(5))
    try:
        _select_hidden_layer(layers, True)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for bool layer_index")


@case("_select_hidden_layer: garbage type raises TypeError")
def t_select_garbage():
    layers = tuple(torch.full((1, 4, 3), float(i)) for i in range(5))
    try:
        _select_hidden_layer(layers, object())
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for non-iterable")


@case("_compute_repr_diagnostic_metrics: writes expected keys")
def t_diag_metrics():
    s, h_dim = 16, 8
    sent_emb = torch.randn(s, h_dim)
    sent_emb = sent_emb / (sent_emb.norm(dim=-1, keepdim=True) + 1e-8)
    sent_score = torch.randn(s)
    sent_correct = torch.zeros(s, dtype=torch.bool)
    sent_correct[:8] = True  # half correct
    metrics = {}
    _compute_repr_diagnostic_metrics(
        sent_emb=sent_emb, sent_score=sent_score, sent_correct_mask=sent_correct,
        metrics=metrics, prefix="slpa",
    )
    expected_keys = {
        "slpa/repr/cos_sim_global_mean",
        "slpa/repr/cos_sim_pos_neg_gap",
        "slpa/repr/adv_signal_var",
        "slpa/repr/adv_signal_snr",
    }
    assert expected_keys.issubset(metrics.keys()), f"missing keys: {expected_keys - metrics.keys()}"
    for k, v in metrics.items():
        assert isinstance(v, float)
        assert v == v  # not NaN


@case("_compute_repr_diagnostic_metrics: skips with <2 sentences")
def t_diag_metrics_small():
    metrics = {}
    sent_emb = torch.randn(1, 4)
    sent_score = torch.tensor([0.5])
    sent_correct = torch.tensor([True])
    _compute_repr_diagnostic_metrics(
        sent_emb=sent_emb, sent_score=sent_score, sent_correct_mask=sent_correct,
        metrics=metrics, prefix="scr",
    )
    assert metrics == {}


@case("_compute_repr_diagnostic_metrics: handles all-correct (no neg) gracefully")
def t_diag_metrics_allcorrect():
    s = 5
    sent_emb = torch.randn(s, 4)
    sent_score = torch.randn(s)
    sent_correct = torch.ones(s, dtype=torch.bool)
    metrics = {}
    _compute_repr_diagnostic_metrics(
        sent_emb=sent_emb, sent_score=sent_score, sent_correct_mask=sent_correct,
        metrics=metrics, prefix="scr",
    )
    # No pos/neg gap should be written.
    assert "scr/repr/cos_sim_pos_neg_gap" not in metrics
    # But the others should be there.
    assert "scr/repr/cos_sim_global_mean" in metrics


@case("SentenceReprConfig: defaults are last-layer + last-token")
def t_repr_config_defaults():
    # Bypass full verl import; load algorithm.py directly. It only needs
    # verl.base_config which itself is light.
    spec = importlib.util.spec_from_file_location(
        "verl.base_config", str(PROJECT_ROOT / "verl/base_config.py")
    )
    bc = importlib.util.module_from_spec(spec)
    sys.modules["verl"] = type(sys)("verl")
    sys.modules["verl.base_config"] = bc
    spec.loader.exec_module(bc)
    spec2 = importlib.util.spec_from_file_location(
        "algorithm", str(PROJECT_ROOT / "verl/trainer/config/algorithm.py")
    )
    algo = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(algo)

    rc = algo.SentenceReprConfig()
    assert rc.hidden_layer_index == -1
    assert rc.pooling == "last"

    # Each adv config has a repr field with defaults.
    sa = algo.SentenceAdvConfig()
    assert sa.repr.hidden_layer_index == -1 and sa.repr.pooling == "last"
    sl = algo.SLPAConfig()
    assert sl.repr.hidden_layer_index == -1 and sl.repr.pooling == "last"
    sc = algo.SCRConfig()
    assert sc.repr.hidden_layer_index == -1 and sc.repr.pooling == "last"

    # Override works.
    sl2 = algo.SLPAConfig(repr=algo.SentenceReprConfig(hidden_layer_index=-9, pooling="mean"))
    assert sl2.repr.hidden_layer_index == -9 and sl2.repr.pooling == "mean"

    # repr field is independent across instances (no shared mutable default).
    sl3 = algo.SLPAConfig()
    assert sl3.repr is not sl.repr


@case("build_punct_token_ids: includes ascii punct, excludes word tokens")
def t_punct():
    class MockTok:
        def __init__(self):
            self.vocab = {"hello": 0, ".": 1, "world": 2, "!": 3, " ": 4, "test": 5}
        def get_vocab(self):
            return self.vocab
        def decode(self, ids, skip_special_tokens=False):
            inv = {v: k for k, v in self.vocab.items()}
            return "".join(inv[i] for i in ids)

    ids = sr.build_punct_token_ids(MockTok())
    s = set(ids.tolist())
    assert 1 in s and 3 in s and 4 in s, f"missing punct ids in {s}"
    assert 0 not in s and 2 not in s and 5 not in s, f"word ids leaked into punct: {s}"


# ---------------------------- main ----------------------------

if __name__ == "__main__":
    tests = [
        t_shapes, t_backcompat_last, t_first, t_mean, t_diff,
        t_mean_no_punct_fallback, t_mean_no_punct_normal, t_entropy_weighted,
        t_invalid_pool, t_empty,
        t_select_int, t_select_list, t_select_oneelem, t_select_empty,
        t_select_listlike, t_select_bool, t_select_garbage,
        t_diag_metrics, t_diag_metrics_small, t_diag_metrics_allcorrect,
        t_repr_config_defaults, t_punct,
    ]
    print(f"Running {len(tests)} cases...\n")
    for t in tests:
        t()
    print(f"\n{'='*48}\nResults: {PASS} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
