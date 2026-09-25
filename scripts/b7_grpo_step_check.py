#!/usr/bin/env python3
"""Minimal GRPO update check (README §4.4-4.5): one group, one optimizer step. A pipeline check, not training.

Order (strict on-policy, README §4.5):
  1. G rollouts with the current LoRA (35-step SDE, a = --a), no grad; cache only the selected steps.
  2. Parameters are asserted unchanged from the start of the rollouts to the end of accumulation.
  3. For each candidate g and each selected step j, one at a time (batch 1):
       reference mean (LoRA off, no grad) -> policy log-prob and mean (LoRA on, grad) ->
       loss_gj = [ -min(rho A_g, clip(rho, 1-eps, 1+eps) A_g) + beta * KL_j ] / (G K) -> backward.
     LoRA-off forwards are refused under grad (lora.LoRALinear), so the policy forward and its checkpointed
     backward cannot silently run with LoRA disabled.
  4. One optimizer.step() (AdamW on the LoRA parameters). No native-supervision update in between.
Advantages are ARTIFICIAL (fixed, standardized) unless a reward is supplied: this checks the optimizer path only
and says nothing about training benefit.
Checks after the step: grads finite, LoRA parameters changed, base weights bitwise unchanged, the policy log-prob
of a cached step moved, and LoRA save -> load into a fresh DiT reproduces the sampled output bitwise.
Reports the peak allocated memory of the whole group (rollouts + cache + reference/policy forwards + backward +
AdamW state) and the wall time per phase.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
AV = ROOT / "third_party" / "AnyView-DVS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AV))

from anyd4rt.anyview_rl import lora, rollout, sde  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("b4", ROOT / "scripts" / "b4_sampler_checks.py")
b4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b4)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="zeroshot:droid_OOD_LR")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--K", type=int, default=12, help="number of denoising steps used for the update (evenly spaced)")
    ap.add_argument("--a", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=1e-4, help="per-dimension-mean clip range (Flow-GRPO default)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "b7"))
    return ap.parse_args()


def build(config, device, rank, seed=0, ckpt=True):
    from anyview.pipe import AnyViewPipeline, load_dit, load_text_emb
    dit = load_dit(config, sac_mode="none").to(device=device, dtype=torch.bfloat16)
    wrapped = lora.inject(dit, rank=rank, alpha=float(rank), seed=seed)
    if ckpt:
        lora.enable_block_checkpoint(dit)
    dit.train()
    return dit, wrapped, AnyViewPipeline(dit, load_text_emb(config), device=str(device), dtype=torch.bfloat16)


def base_checksums(dit):
    return torch.stack([p.detach().double().sum() for n, p in dit.named_parameters() if "lora_" not in n])


def main():
    a = parse_args()
    from anyview.avb_dataset import AVBDataset
    from anyview.config import AnyViewConfig
    from anyview.vae import load_vae

    device = torch.device("cuda")
    config = AnyViewConfig(checkpoint_path=str(AV / "checkpoints/anyview_dvs_2b.pt"), tokenizer_path=str(AV / "checkpoints/tokenizer.pth"))
    vae = load_vae(config.tokenizer_path, device)
    dit, wrapped, pipe = build(config, device, a.rank)
    params = lora.lora_parameters(dit)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)

    tier, split = a.split.split(":")
    item = AVBDataset(str(AV / "data" / f"AnyViewBench_{tier}"), split)[a.episode]
    with torch.no_grad():
        entries, _ = b4.make_entries(item, vae, config, device)
        _, masks, _, _ = rollout._setup(pipe, entries, 0, 35)
    om = masks["output"]["v0"].bool()
    sel = sorted(set(np.linspace(0, 34, a.K).round().astype(int).tolist()))
    adv = torch.tensor(np.linspace(1, -1, a.G), dtype=torch.float64)
    adv = ((adv - adv.mean()) / (adv.std() + 1e-4)).clamp(-5, 5)  # ARTIFICIAL advantages (optimizer-path check only)

    base0 = base_checksums(dit)
    lora0 = [p.detach().clone() for p in params]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 2**30
    t0 = time.time()

    # 1. rollouts (no grad), cache only the selected steps
    cache = []
    for g in range(a.G):
        r = rollout.sample(pipe, entries, seed=g, num_steps=35, mode="sde", a=a.a, noise_seed=g, record=True)
        cache.append([s for s in r.steps if s["i"] in sel])
        del r
    torch.cuda.synchronize()
    t_rollout = time.time() - t0
    mem_after_rollout = torch.cuda.max_memory_allocated() / 2**30
    cache_GiB = sum(s["x"].numel() * 4 + s["z_next"].numel() * 8 for c in cache for s in c) / 2**30

    # 2-3. accumulation, one (candidate, step) at a time
    t1 = time.time()
    ratios, kls, losses = [], [], []
    for g in range(a.G):
        for rec in cache[g]:
            lora.set_enabled(dit, False)
            _, mu_ref = rollout.recompute_log_prob(pipe, entries, rec, a=a.a)  # reference: LoRA off, no grad
            lora.set_enabled(dit, True)
            with torch.enable_grad():
                lp, mu = rollout.step_log_prob(pipe, entries, rec, a=a.a)
                rho = torch.exp(lp - rec["logp_old"])
                surr = -torch.minimum(rho * adv[g], rho.clamp(1 - a.eps, 1 + a.eps) * adv[g])
                kl = sde.kl(mu, mu_ref, rec["std"], mask=om)
                loss = (surr + a.beta * kl).mean() / (a.G * len(cache[g]))
                loss.backward()
            ratios.append(float(rho.detach()[0])); kls.append(float(kl.detach()[0])); losses.append(float(loss.detach()))
            del mu, mu_ref, lp
    torch.cuda.synchronize()
    t_accum = time.time() - t1
    unchanged = all(torch.equal(p.detach(), q) for p, q in zip(params, lora0))

    grads = [p.grad for p in params]
    grads_finite = all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    grad_norm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads if g is not None)))
    t2 = time.time()
    opt.step()
    torch.cuda.synchronize()
    t_step = time.time() - t2
    peak = torch.cuda.max_memory_allocated() / 2**30
    t_group = time.time() - t0
    opt_state_GiB = sum(v.numel() * v.element_size() for st in opt.state.values() for v in st.values() if torch.is_tensor(v)) / 2**30

    changed = [not torch.equal(p.detach(), q) for p, q in zip(params, lora0)]
    base_same = bool(torch.equal(base_checksums(dit), base0))
    lp_after, _ = rollout.recompute_log_prob(pipe, entries, cache[0][0], a=a.a)
    moved = float((lp_after - cache[0][0]["logp_old"]).abs()[0])

    # LoRA save -> load into a fresh DiT (different init seed, no checkpointing) -> same sampled output
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(lora.lora_state_dict(dit), out / "lora_after_step.pt")
    ref_out = rollout.sample(pipe, entries, seed=0, num_steps=35, mode="sde", a=a.a, noise_seed=0).y0_pred_streams["v0"][:, :16].float()
    del cache
    dit2, _, pipe2 = build(config, device, a.rank, seed=123, ckpt=False)
    lora.load_lora_state_dict(dit2, torch.load(out / "lora_after_step.pt"))
    re_out = rollout.sample(pipe2, entries, seed=0, num_steps=35, mode="sde", a=a.a, noise_seed=0).y0_pred_streams["v0"][:, :16].float()
    lora.set_enabled(dit2, False)
    base_out = rollout.sample(pipe2, entries, seed=0, num_steps=35, mode="sde", a=a.a, noise_seed=0).y0_pred_streams["v0"][:, :16].float()

    res = {
        "args": vars(a), "selected_steps": sel, "advantages_artificial": adv.tolist(),
        "latent_rgb0_shape": list(entries["rgb0"].shape),
        "params_unchanged_rollout_to_step": unchanged,
        "ratio_range": [min(ratios), max(ratios)], "kl_range": [min(kls), max(kls)],
        "grads_finite": grads_finite, "grad_norm": grad_norm,
        "lora_tensors_changed_frac": sum(changed) / len(changed),
        "base_weights_bitwise_unchanged": base_same,
        "logp_change_after_step_on_cached_step0": moved,
        "reload_output_max_abs_diff": float((re_out - ref_out).abs().max()),
        "updated_vs_base_output_max_abs_diff": float((ref_out - base_out).abs().max()),
        "time_s": {"rollouts": t_rollout, "accumulation": t_accum, "optimizer_step": t_step, "group_total": t_group,
                   "per_policy_fwd_bwd_incl_ref": t_accum / len(ratios)},
        "memory_GiB": {"allocated_before_group": mem_before, "peak_after_rollouts": mem_after_rollout,
                       "rollout_cache": cache_GiB, "adamw_state": opt_state_GiB, "peak_group_total": peak},
        "n_policy_terms": len(ratios),
    }
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print("B7_RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
