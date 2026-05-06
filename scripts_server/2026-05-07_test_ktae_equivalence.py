"""Element-wise equivalence test: my port vs KTAE upstream.

Imports both the upstream module (~/KTAE/verl/...) and our ported module
(~/sentencepo_v1-5/verl/...) into the same process, runs both on identical
synthetic batches, and asserts torch.allclose on the output advantage tensor.

This is the gold-standard verification that the algorithm is preserved
modulo the documented intentional changes (pad_token_id parameterisation;
defaults match upstream actual runtime values).

Run on CPU (no GPU on this node).
"""
import importlib
import os
import sys
import types

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

# We must import upstream's compute_key_tokens first (without our import-path
# pollution), then ours. To avoid module-name collisions we load each module
# directly from file using importlib.util.

import importlib.util


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


UPSTREAM = os.path.expanduser("~/KTAE")
OURS = os.path.expanduser("~/sentencepo_v1-5")

# Load upstream ComputeKeyTokens directly (no verl package needed).
upstream_ckt = load_module("up_ckt", f"{UPSTREAM}/verl/workers/actor/compute_key_tokens.py")
ours_ckt = load_module("our_ckt", f"{OURS}/verl/workers/actor/compute_key_tokens.py")


def build_batch(g=8, t=64, vocab=151700, n_correct=3, signal=42, antisignal=43):
    """Realistic Qwen3-scale vocab so upstream's hardcoded ``[151643] = 0`` is in-bounds.

    Plants the Qwen pad id (151643) somewhere in the batch so ``responses.max()``
    is at least 151643, making the upstream key_token_result array large enough.
    """
    responses = torch.randint(0, vocab, (g, t), dtype=torch.long)
    # Plant signal at three positions that scale to t (avoids out-of-bounds for short t).
    sig_pos = [t // 8, t // 4, t // 2]
    responses[:n_correct, sig_pos] = signal
    responses[n_correct:, sig_pos] = antisignal
    # Make sure the Qwen pad id appears at least once so responses.max() >= 151643
    responses[0, -1] = 151643
    mask = torch.ones(g, t, dtype=torch.float32)
    mask[:, -3:] = 0.0  # ragged tail
    rewards = torch.zeros(g, dtype=torch.float32)
    rewards[:n_correct] = 1.0
    tlr = torch.zeros(g, t, dtype=torch.float32)
    tlr[:, -4] = rewards  # scalar reward at fixed position
    index = np.array(["q0"] * g, dtype=object)
    return tlr, mask, responses, index, rewards


def upstream_ktae(token_level_rewards, responses, eos_mask, index, epsilon=1e-6):
    """Verbatim copy of upstream compute_ktae_outcome_advantage_and_keytokens, with the
    upstream ComputeKeyTokens (no pad_token_id param)."""
    from collections import defaultdict

    response_length = token_level_rewards.shape[-1]
    id2score = defaultdict(list)
    id2reponses = {}
    id2mask = {}
    id2mean = {}
    id2std = {}
    scores = token_level_rewards.sum(dim=-1)
    with torch.no_grad():
        bsz = token_level_rewards.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            if index[i] in id2reponses:
                id2reponses[index[i]] = torch.cat((id2reponses[index[i]], responses[i].unsqueeze(0)), dim=0)
            else:
                id2reponses[index[i]] = responses[i].unsqueeze(0)
            if index[i] in id2mask:
                id2mask[index[i]] = torch.cat((id2mask[index[i]], eos_mask[i].unsqueeze(0)), dim=0)
            else:
                id2mask[index[i]] = eos_mask[i].unsqueeze(0)
        id2key_token = {}
        for idx in id2score:
            reponses_per_q = id2reponses[idx]
            mask_per_q = id2mask[idx]
            score_per_q = id2score[idx]
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
                id2key_token[idx] = torch.zeros([responses.max().item()], device=responses.device)
            elif len(id2score[idx]) > 1:
                format_score_per_q = torch.tensor(score_per_q)
                id2mean[idx] = torch.mean(format_score_per_q)
                id2std[idx] = torch.std(format_score_per_q)
                computer = upstream_ckt.ComputeKeyTokens(
                    alpha=1.0,
                    beta_ig=1.0,
                    gamma_tf=1.0,
                    top=1.0,
                    bottom=-1.0,
                    responses_ids=reponses_per_q,
                    mask=mask_per_q,
                    rewards=format_score_per_q,
                    max_token_num=responses.max().item(),
                )
                key_tokens = computer.get_key_tokens().to(reponses_per_q.device)
                id2key_token[idx] = key_tokens
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        means = torch.tensor([id2mean[index[i]] for i in range(bsz)], device=scores.device)
        stds = torch.tensor([id2std[index[i]] for i in range(bsz)], device=scores.device)
        scores = (scores - means) / (stds + epsilon)
        format_weights = [id2key_token[index[i]][responses[i]].unsqueeze(0) for i in range(bsz)]
        all_weight = torch.cat(format_weights, dim=0)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask + all_weight * eos_mask
    return scores, scores


# Now load OUR registered function via the verl package
sys.path.insert(0, OURS)
# Drop any stale modules from upstream load
for k in list(sys.modules):
    if k.startswith("verl"):
        del sys.modules[k]
from verl.trainer.config.algorithm import AlgoConfig, KTAEConfig  # noqa: E402
from verl.trainer.ppo import core_algos as ca  # noqa: E402

ours_fn = ca.get_adv_estimator_fn("ktae")


def run_one(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tlr, mask, responses, index, rewards = build_batch()

    up_adv, _ = upstream_ktae(tlr, responses, mask, index)

    cfg = AlgoConfig(adv_estimator="ktae", ktae=KTAEConfig())  # defaults: 1/1/1/1/-1, pad=151643
    our_adv, _ = ours_fn(
        token_level_rewards=tlr,
        response_mask=mask,
        index=index,
        responses=responses,
        config=cfg,
    )

    same = torch.allclose(up_adv, our_adv, atol=1e-6, rtol=0)
    max_abs = (up_adv - our_adv).abs().max().item()
    print(
        f"seed={seed}: allclose={same}  max_abs_diff={max_abs:.2e}  "
        f"up.mean={up_adv.mean().item():+.4f}  our.mean={our_adv.mean().item():+.4f}"
    )
    return same


print("=== KTAE port equivalence test ===")
all_ok = True
print("\n[A] basic 5-seed comparison (G=8, T=64, vocab~Qwen3, balanced 3 vs 5):")
for s in [0, 1, 2, 7, 42]:
    all_ok &= run_one(s)


def run_edge(name, **kw):
    """Run one edge-case batch through both implementations."""
    torch.manual_seed(123)
    np.random.seed(123)
    tlr, mask, responses, index, rewards = build_batch(**kw)
    up_adv, _ = upstream_ktae(tlr, responses, mask, index)
    cfg = AlgoConfig(adv_estimator="ktae", ktae=KTAEConfig())
    our_adv, _ = ours_fn(
        token_level_rewards=tlr,
        response_mask=mask,
        index=index,
        responses=responses,
        config=cfg,
    )
    same = torch.allclose(up_adv, our_adv, atol=1e-6, rtol=0)
    max_abs = (up_adv - our_adv).abs().max().item()
    print(f"  {name:36s}  allclose={same}  max_abs_diff={max_abs:.2e}")
    return same


print("\n[B] edge cases:")
all_ok &= run_edge("balanced (3 correct / 5 wrong)", g=8, n_correct=3)
all_ok &= run_edge("only 1 correct out of 8", g=8, n_correct=1)
all_ok &= run_edge("only 1 wrong out of 8 (7 correct)", g=8, n_correct=7)
all_ok &= run_edge("very long response (T=512)", g=8, t=512, n_correct=3)
all_ok &= run_edge("short response (T=16)", g=8, t=16, n_correct=3)
all_ok &= run_edge("larger group G=16", g=16, n_correct=5)


def run_multi_group():
    """2 groups of size 8 in one batch (different uids), each with own correctness pattern."""
    torch.manual_seed(7)
    np.random.seed(7)
    g, t = 8, 64
    vocab = 151700
    bsz = 2 * g
    responses = torch.randint(0, vocab, (bsz, t), dtype=torch.long)
    # group A: 3 correct out of 8, signal token 100 in correct
    responses[:3, [5, 17, 29]] = 100
    responses[3:8, [5, 17, 29]] = 101
    # group B: 5 correct out of 8, different signal token 200
    responses[8:13, [5, 17, 29]] = 200
    responses[13:16, [5, 17, 29]] = 201
    responses[0, -1] = 151643  # ensure max id
    mask = torch.ones(bsz, t, dtype=torch.float32)
    mask[:, -3:] = 0.0
    rewards = torch.zeros(bsz, dtype=torch.float32)
    rewards[:3] = 1.0
    rewards[8:13] = 1.0
    tlr = torch.zeros(bsz, t, dtype=torch.float32)
    tlr[:, -4] = rewards
    index = np.array(["qA"] * g + ["qB"] * g, dtype=object)

    up_adv, _ = upstream_ktae(tlr, responses, mask, index)
    cfg = AlgoConfig(adv_estimator="ktae", ktae=KTAEConfig())
    our_adv, _ = ours_fn(
        token_level_rewards=tlr,
        response_mask=mask,
        index=index,
        responses=responses,
        config=cfg,
    )
    same = torch.allclose(up_adv, our_adv, atol=1e-6, rtol=0)
    max_abs = (up_adv - our_adv).abs().max().item()
    print(f"  {'2 groups in one batch':36s}  allclose={same}  max_abs_diff={max_abs:.2e}")
    return same


print("\n[C] multi-group:")
all_ok &= run_multi_group()


def run_hyperparam_match():
    """Verify ablation hyperparameter combinations are still bit-exact."""
    torch.manual_seed(42)
    np.random.seed(42)
    tlr, mask, responses, index, _ = build_batch()
    cases = [
        (1.0, 1.0, 1.0),  # default upstream runtime
        (2.0, 1.0, 1.0),  # alpha higher
        (1.0, 2.0, 1.0),  # beta_ig higher (matches the dead-config shell)
        (0.0, 1.0, 1.0),  # disable Fisher channel (alpha=0 in upstream)
        (1.0, 0.0, 1.0),  # disable IG channel (beta_ig=0 in upstream)
        (1.0, 1.0, 0.0),  # disable TF correction (gamma_tf=0)
    ]
    ok = True
    for alpha, beta_ig, gamma_tf in cases:
        # Patch upstream's hardcoded values by monkey-patching the call inside upstream_ktae.
        global upstream_ktae

        def upstream_with_hyper(token_level_rewards, responses, eos_mask, index, epsilon=1e-6):
            from collections import defaultdict
            response_length = token_level_rewards.shape[-1]
            id2score = defaultdict(list)
            id2reponses = {}
            id2mask = {}
            id2mean = {}
            id2std = {}
            scores = token_level_rewards.sum(dim=-1)
            with torch.no_grad():
                bsz = token_level_rewards.shape[0]
                for i in range(bsz):
                    id2score[index[i]].append(scores[i])
                    id2reponses[index[i]] = (
                        torch.cat((id2reponses[index[i]], responses[i].unsqueeze(0)), dim=0)
                        if index[i] in id2reponses else responses[i].unsqueeze(0)
                    )
                    id2mask[index[i]] = (
                        torch.cat((id2mask[index[i]], eos_mask[i].unsqueeze(0)), dim=0)
                        if index[i] in id2mask else eos_mask[i].unsqueeze(0)
                    )
                id2key_token = {}
                for idx in id2score:
                    reponses_per_q = id2reponses[idx]
                    mask_per_q = id2mask[idx]
                    score_per_q = id2score[idx]
                    if len(id2score[idx]) > 1:
                        format_score_per_q = torch.tensor(score_per_q)
                        id2mean[idx] = torch.mean(format_score_per_q)
                        id2std[idx] = torch.std(format_score_per_q)
                        computer = upstream_ckt.ComputeKeyTokens(
                            alpha=alpha, beta_ig=beta_ig, gamma_tf=gamma_tf,
                            top=1.0, bottom=-1.0,
                            responses_ids=reponses_per_q, mask=mask_per_q,
                            rewards=format_score_per_q, max_token_num=responses.max().item(),
                        )
                        id2key_token[idx] = computer.get_key_tokens().to(reponses_per_q.device)
                    else:
                        id2mean[idx] = torch.tensor(0.0)
                        id2std[idx] = torch.tensor(1.0)
                        id2key_token[idx] = torch.zeros([responses.max().item()], device=responses.device)
                means = torch.tensor([id2mean[index[i]] for i in range(bsz)], device=scores.device)
                stds = torch.tensor([id2std[index[i]] for i in range(bsz)], device=scores.device)
                scores = (scores - means) / (stds + epsilon)
                format_weights = [id2key_token[index[i]][responses[i]].unsqueeze(0) for i in range(bsz)]
                all_weight = torch.cat(format_weights, dim=0)
                scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask + all_weight * eos_mask
            return scores, scores

        up_adv, _ = upstream_with_hyper(tlr, responses, mask, index)
        cfg = AlgoConfig(
            adv_estimator="ktae",
            ktae=KTAEConfig(alpha=alpha, beta_ig=beta_ig, gamma_tf=gamma_tf),
        )
        our_adv, _ = ours_fn(
            token_level_rewards=tlr,
            response_mask=mask,
            index=index,
            responses=responses,
            config=cfg,
        )
        same = torch.allclose(up_adv, our_adv, atol=1e-6, rtol=0)
        max_abs = (up_adv - our_adv).abs().max().item()
        label = f"a={alpha} bIG={beta_ig} gTF={gamma_tf}"
        print(f"  {label:36s}  allclose={same}  max_abs_diff={max_abs:.2e}")
        ok &= same
    return ok


print("\n[D] hyperparameter sweeps (each cfg compared to upstream-with-same-hyper):")
all_ok &= run_hyperparam_match()

print("\nOVERALL:", "PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
