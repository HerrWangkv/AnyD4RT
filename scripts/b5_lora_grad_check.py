#!/usr/bin/env python3
"""B5 (README §6 S3): gradient-enabled LoRA recomputation of the SDE log-prob, minimal check + memory.

For one episode, with a DiT built with the given activation-checkpointing mode and LoRA injected:
  G1 identity:     LoRA with B = 0 enabled vs disabled -> same log-prob (the injected model is the released one).
  G2 grad vs no-grad: log-prob recomputed under enable_grad vs no_grad, same parameters -> difference.
  G3 gradients:    backward of the mean log-prob: which LoRA tensors get finite, nonzero gradients
                   (at B = 0 only B gets a gradient; after a small random B, A as well).
  G4 memory/time:  peak allocated memory and wall time of one forward + backward (batch 1), after warm-up.
The rollout itself (35-step SDE, recorded) runs under no_grad with the same injected model (B = 0).
Resolution is the released snap (long side 576) unless --target sets a smaller long side.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
AV = ROOT / "third_party" / "AnyView-DVS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AV))

from anyd4rt.anyview_rl import lora, rollout  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("b4", ROOT / "scripts" / "b4_sampler_checks.py")
b4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b4)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="zeroshot:droid_OOD_LR")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--sac-mode", default="none", choices=("none", "mm_only", "block_wise"))
    ap.add_argument("--block-ckpt", action="store_true", help="our full per-block recomputation (released block_wise mode is broken)")
    ap.add_argument("--save-grads", default="", help="save the LoRA B gradients of the first checked step to this .pt file")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--target", type=int, default=576, help="long side of the inference grid (released: 576)")
    ap.add_argument("--num-frames", type=int, default=41)
    ap.add_argument("--a", type=float, default=0.7)
    ap.add_argument("--steps-to-check", default="0,17,34")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "b5"))
    return ap.parse_args()


def main():
    a = parse_args()
    from anyview.avb_dataset import AVBDataset
    from anyview.config import AnyViewConfig
    from anyview.pipe import AnyViewPipeline, load_dit, load_text_emb
    from anyview.vae import load_vae

    device = torch.device("cuda")
    config = AnyViewConfig(checkpoint_path=str(AV / "checkpoints/anyview_dvs_2b.pt"), tokenizer_path=str(AV / "checkpoints/tokenizer.pth"))
    vae = load_vae(config.tokenizer_path, device)
    dit = load_dit(config, sac_mode=a.sac_mode).to(device=device, dtype=torch.bfloat16)
    wrapped = lora.inject(dit, rank=a.rank, alpha=float(a.rank))
    if a.block_ckpt:
        assert a.sac_mode == "none"
        lora.enable_block_checkpoint(dit)
    dit.train()  # training-mode recomputation; B4 found train() == eval() for this DiT (no active dropout)
    pipe = AnyViewPipeline(dit, load_text_emb(config), device=str(device), dtype=torch.bfloat16)
    n_lora = sum(p.numel() for p in lora.lora_parameters(dit))

    tier, split = a.split.split(":")
    item = AVBDataset(str(AV / "data" / f"AnyViewBench_{tier}"), split)[a.episode]
    if a.num_frames != 41:
        for k in ("input", "target_gt", "intrinsics_input", "intrinsics_target", "world2cam_input", "world2cam_target"):
            if torch.is_tensor(item[k]) and item[k].ndim >= 3 and item[k].shape[0] >= a.num_frames:
                item[k] = item[k][: a.num_frames]
    if a.target != 576:  # smaller inference grid: same released snapping rule with a different long side
        orig = b4.infer_lib.snap_shape
        b4.infer_lib.snap_shape = lambda h, w, target=576, stride=16, grid_res=384: orig(h, w, target=a.target, stride=stride, grid_res=min(grid_res, a.target))
    with torch.no_grad():
        entries, _ = b4.make_entries(item, vae, config, device)
    latent_shape = tuple(entries["rgb0"].shape)

    # Rollout with the injected model (B = 0), recorded.
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    r = rollout.sample(pipe, entries, seed=0, num_steps=35, mode="sde", a=a.a, noise_seed=0, record=True)
    rollout_s, rollout_peak = time.time() - t0, torch.cuda.max_memory_allocated() / 2**30
    recs = [r.steps[int(j)] for j in a.steps_to_check.split(",")]
    res = {"args": vars(a), "latent_rgb0_shape": latent_shape, "n_lora_modules": len(wrapped), "n_lora_params": n_lora,
           "rollout_35_steps_s": rollout_s, "rollout_peak_alloc_GiB": rollout_peak, "steps": []}

    for rec in recs:
        row = {"i": rec["i"], "logp_old": float(rec["logp_old"][0])}
        lora.set_enabled(dit, False)
        lp_off, _ = rollout.recompute_log_prob(pipe, entries, rec, a=a.a)
        lora.set_enabled(dit, True)
        lp_nog, _ = rollout.recompute_log_prob(pipe, entries, rec, a=a.a)
        row["G1_lora_B0_on_vs_off"] = float((lp_nog - lp_off).abs()[0])
        row["rollout_vs_recompute_nograd"] = float((lp_nog - rec["logp_old"]).abs()[0])

        for p in lora.lora_parameters(dit):
            p.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated() / 2**30
        t1 = time.time()
        with torch.enable_grad():
            lp, _ = rollout.step_log_prob(pipe, entries, rec, a=a.a)
            lp.mean().backward()
        torch.cuda.synchronize()
        row["fwd_bwd_s"] = time.time() - t1
        row["fwd_bwd_peak_alloc_GiB"] = torch.cuda.max_memory_allocated() / 2**30
        row["alloc_before_GiB"] = base_alloc
        row["G2_grad_vs_nograd"] = float((lp.detach() - lp_nog).abs()[0])
        gA = [m.lora_A.grad for m in wrapped]
        gB = [m.lora_B.grad for m in wrapped]
        row["G3_B0"] = {"B_grad_nonzero_frac": sum(int(g is not None and bool(g.abs().sum() > 0)) for g in gB) / len(gB),
                        "B_grad_finite": all(g is not None and bool(torch.isfinite(g).all()) for g in gB),
                        "A_grad_nonzero_frac": sum(int(g is not None and bool(g.abs().sum() > 0)) for g in gA) / len(gA),
                        "B_grad_norm": float(torch.sqrt(sum((g.float() ** 2).sum() for g in gB if g is not None)))}
        if a.save_grads and rec is recs[0]:
            torch.save(torch.cat([g.flatten().float().cpu() for g in gB]), a.save_grads)
        res["steps"].append(row)
        print(json.dumps(row), flush=True)

    # G3 with a small random B: A must receive gradients too.
    with torch.no_grad():
        g = torch.Generator(device=device).manual_seed(0)
        for m in wrapped:
            m.lora_B.copy_(1e-4 * torch.randn(m.lora_B.shape, generator=g, device=device))
    for p in lora.lora_parameters(dit):
        p.grad = None
    with torch.enable_grad():
        lp, _ = rollout.step_log_prob(pipe, entries, recs[0], a=a.a)
        lp.mean().backward()
    res["G3_smallB"] = {"A_grad_nonzero_frac": sum(int(m.lora_A.grad is not None and bool(m.lora_A.grad.abs().sum() > 0)) for m in wrapped) / len(wrapped),
                        "A_grad_finite": all(m.lora_A.grad is not None and bool(torch.isfinite(m.lora_A.grad).all()) for m in wrapped),
                        "logp_change_vs_B0": float((lp.detach() - recs[0]["logp_old"]).abs()[0])}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{split}_ep{a.episode}_sac-{'blockckpt' if a.block_ckpt else a.sac_mode}_t{a.target}_f{a.num_frames}"
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print("B5_RESULT " + json.dumps({k: v for k, v in res.items() if k != "steps"}, default=float))


if __name__ == "__main__":
    main()
