#!/usr/bin/env python3
"""Small AnyView validation on the real DynamicReplica stereo clips.

This script is deliberately an inference/evaluation harness, not a training
entry point.  It builds the unified AnyView episode from the synchronized
DynamicReplica annotations, generates target-view videos from the real left
video, and scores the same query pool with the frozen OpenD4RT critic under
the unfiltered (N) and source-term-filtered (S) masks.

The D4RT target GT and visibility are the same left-world-trajectory/right-
camera construction used by ``dynamic_replica_stereo_calibration.py``.  The
right RGB video is retained only for auxiliary image and failure-mode checks.
AnyView requires ``1 + 4k`` frames, so generation uses 41 frames (the
calibration reports use the first 40 frames); the critic and all geometry
metrics below use the common first 40 frames.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
OD_ROOT = ROOT / "third_party" / "Open-d4rt"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(OD_ROOT) not in sys.path:
    sys.path.insert(0, str(OD_ROOT))

from dynamic_replica_stereo_calibration import (  # noqa: E402
    LOCKED_THRESHOLDS,
    P,
    T_D4RT,
    _load_annotations,
    _load_clip,
    _path,
    _per_anchor,
    _predict_view,
    _run_critic,
    _selected_stats,
    _source_support,
    _viewpoint_to_camera,
)
from anyd4rt.d4rt_critic import load_d4rt  # noqa: E402
from anyd4rt.reward_d4rt_gt import fit_scale, reward_d4rt_gt  # noqa: E402


T_ANYVIEW = 41


def _frame_name(index: int) -> str:
    return f"{index:010d}"


def _write_link(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() and dst.resolve() == src.resolve():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _build_episode(root: Path, split: str, scene: str, start: int, stride: int,
                   annotations: dict, out_dir: Path, scale_factor: float) -> dict:
    """Create a self-contained-by-reference AnyView unified episode."""
    cams = annotations[scene]
    frame_ids = [int(start + stride * i) for i in range(T_ANYVIEW)]
    if not all(fid in cams[view] for view in ("left", "right") for fid in frame_ids):
        raise ValueError(f"{scene}: missing synchronized frames for 41-frame AnyView episode")

    first = cams["left"][frame_ids[0]]
    image_h, image_w = map(int, first["image"]["size"])
    for view, cam_name in (("cam1", "left"), ("cam0", "right")):
        for index, fid in enumerate(frame_ids):
            item = cams[cam_name][fid]
            h, w = map(int, item["image"]["size"])
            if (h, w) != (image_h, image_w):
                raise ValueError(f"{scene} {cam_name} resolution changed at frame {fid}: {(h, w)}")
            image_path = _path(root, split, item, "image")
            if image_path is None or not image_path.exists():
                raise FileNotFoundError(f"missing {cam_name} RGB for {scene} frame {fid}: {image_path}")
            _write_link(out_dir / "rgb" / view / f"{_frame_name(index)}.png", image_path)

            K, T_cw, _ = _viewpoint_to_camera(item["viewpoint"], image_h, image_w,
                                               "dynamic_replica_v2")
            c2w = np.linalg.inv(T_cw).astype(np.float32)
            lowdim_path = out_dir / "lowdim" / view / f"{_frame_name(index)}.npz"
            lowdim_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(lowdim_path, camera=view, timestep=np.int64(fid),
                     intrinsics=K.astype(np.float32), extrinsics=c2w)

    metadata = {
        "info": {"name": f"dynamic_replica_{scene}", "storage": "frames"},
        "cameras": ["cam0", "cam1"],
        "resolution": [image_h, image_w],
        "num_frames": T_ANYVIEW,
        "framerate": 10.0,
        "rgb": {"extension": "png"},
        "extrinsics": {"transform": "cam2world"},
        "specific": {
            "roles": {"input": "cam1", "target": "cam0"},
            "scale_factor": float(scale_factor),
            "source": {"dataset": "dynamic_replica", "scene": scene},
            "source_protocol": "DynamicReplica v2 viewpoint converted with dynamic_replica_v2",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"frame_ids": frame_ids, "native_hw": [image_h, image_w], "episode": str(out_dir)}


def _read_generated_frames(out_dir: Path) -> np.ndarray:
    paths = sorted((out_dir / "frames").glob("*.png"))
    if len(paths) < T_D4RT:
        raise FileNotFoundError(f"{out_dir}: expected at least {T_D4RT} generated frames, found {len(paths)}")
    return np.stack([np.asarray(Image.open(path).convert("RGB")) for path in paths[:T_D4RT]], axis=0)


def _resize_video(frames: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    return np.stack([cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
                     for frame in frames], axis=0).astype(np.uint8)


def _load_masks(root: Path, split: str, scene: str, frame_ids: list[int], annotations: dict) -> np.ndarray:
    masks = []
    for fid in frame_ids[:T_D4RT]:
        item = annotations[scene]["right"][fid]
        path = _path(root, split, item, "mask")
        if path is None or not path.exists():
            raise FileNotFoundError(f"right object mask missing for {scene} frame {fid}: {path}")
        masks.append(np.asarray(Image.open(path)) > 0)
    return np.stack(masks, axis=0)


def _psnr(mse: float) -> float | None:
    if mse <= 0:
        return None
    return float(10.0 * np.log10(1.0 / mse))


def _rgb_diagnostics(pred: np.ndarray, gt: np.ndarray, fg: np.ndarray) -> dict:
    """RGB-only diagnostics; they are not used as a 3D correctness label."""
    gt = _resize_video(gt, tuple(pred.shape[1:3]))
    fg = np.stack([cv2.resize(mask.astype(np.uint8), (pred.shape[2], pred.shape[1]),
                              interpolation=cv2.INTER_NEAREST).astype(bool) for mask in fg], axis=0)
    pred_f = pred.astype(np.float32) / 255.0
    gt_f = gt.astype(np.float32) / 255.0
    diff = np.abs(pred_f - gt_f).mean(axis=-1)
    sq = (pred_f - gt_f) ** 2

    def mean_where(values: np.ndarray, mask: np.ndarray) -> float | None:
        return float(values[mask].mean()) if mask.any() else None

    fg_px = fg
    bg_px = ~fg
    frame_mse = sq.mean(axis=(1, 2, 3))
    pred_delta = np.abs(pred_f[1:] - pred_f[:-1]).mean(axis=-1)
    gt_delta = np.abs(gt_f[1:] - gt_f[:-1]).mean(axis=-1)
    temporal_delta_diff = np.abs(pred_delta - gt_delta)
    fg_temporal = fg[1:] | fg[:-1]
    bg_temporal = ~fg_temporal
    return {
        "psnr_db": _psnr(float(sq.mean())),
        "mean_l1": float(diff.mean()),
        "foreground_l1": mean_where(diff, fg_px),
        "background_l1": mean_where(diff, bg_px),
        "foreground_area_fraction": float(fg_px.mean()),
        "temporal_motion_l1": float(temporal_delta_diff.mean()),
        "foreground_temporal_motion_l1": mean_where(temporal_delta_diff, fg_temporal),
        "background_temporal_motion_l1": mean_where(temporal_delta_diff, bg_temporal),
        "frame_mse_p90": float(np.percentile(frame_mse, 90)),
    }


def _compact_stats(stats: dict) -> dict:
    terms = {}
    for term, value in stats["coverage"].items():
        terms[term] = {
            "mean_entry": value["mean_entry"],
            "mean_anchor": value["mean_anchor"],
            "anchor_coverage": value["anchor_coverage"],
            "dynamic_anchor_coverage": value["dynamic_anchor_coverage"],
            "static_anchor_coverage": value["static_anchor_coverage"],
            "dynamic_mean_anchor": value["dynamic"]["mean_anchor"],
            "static_mean_anchor": value["static"]["mean_anchor"],
        }
    random = {}
    for term, controls in stats["equal_anchor_coverage"].items():
        random[term] = {}
        for name, value in controls.items():
            random[term][name] = {
                "n_anchors": value["n_anchors"],
                "n_dynamic_anchors": value["n_dynamic_anchors"],
                "n_static_anchors": value["n_static_anchors"],
                "mean_anchor": value["mean_anchor"],
                "std_mean_anchor": value["std_mean_anchor"],
                "dynamic_mean_anchor": value["dynamic"]["mean_anchor"],
                "static_mean_anchor": value["static"]["mean_anchor"],
                "successful_draws": value["successful_draws"],
                "exact_time_count_match": value["exact_time_count_match"],
            }
    return {"coverage": terms, "matched_random": random}


def _combined_error(stats: dict, field: str) -> float | None:
    values = [stats["coverage"][term][field] for term in ("e", "d", "loc")]
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _support_summary(support: dict, dynamic: np.ndarray) -> dict:
    out = {}
    for term in ("e", "d", "loc"):
        mask = np.asarray(support[term], bool)
        out[term] = {
            "anchors": int(mask.sum()),
            "fraction": float(mask.mean()),
            "dynamic_anchors": int((mask & dynamic).sum()),
            "static_anchors": int((mask & ~dynamic).sum()),
        }
    out["low_motion_dynamic_anchors"] = int(np.asarray(support["low_motion_dynamic"], bool).sum())
    return out


def _critic_target_view(model, clip: dict) -> dict:
    """Run exactly the calibration critic for only the generated target view."""
    view = "right"
    h, w = clip["frames"][view].shape[1:3]
    uv = clip["px"][view][0] * np.array([256.0 / w, 256.0 / h], dtype=np.float32) / 255.0
    pred, pred_loc = _predict_view(model, clip["frames"][view], uv)
    V = clip["visible"][view].T
    fit = fit_scale(pred, clip["Y"][view], V, 0, P)
    out = reward_d4rt_gt(
        pred, clip["Y"][view], V, 0, P, pred_loc, clip["Y_loc"][view],
        term_masks={"e": V, "d": V, "loc": V}, scale_mask=V,
        scale_override=fit["s"] if fit["scale_valid"] else np.nan,
    )
    if not out.get("scale_valid", False):
        raise RuntimeError(f"target scale fit failed for {clip['split']} {clip['scene']}: {fit}")
    return {
        "fit": {key: (float(value) if isinstance(value, (float, np.floating)) else int(value))
                for key, value in fit.items() if key in ("s", "n_Q0", "n_Vplus")},
        "V": V, "Y": clip["Y"][view], "e": out["e"], "d": out["d"], "loc": out["e_loc"],
        "e_i": _per_anchor(out["e"], V),
        "d_i": _per_anchor(out["d"], V & (np.arange(T_D4RT) != 0)[None] & V[:, :1]),
        "loc_i": _per_anchor(out["e_loc"], V & (np.arange(T_D4RT) != 0)[None]),
    }


def _score_candidate(model, clip: dict, generated_raw: np.ndarray | None,
                     source_support: dict, angle_thresholds: dict[str, float], seed: int,
                     predicted_target: dict | None = None) -> dict:
    eval_clip = dict(clip)
    eval_clip["frames"] = dict(clip["frames"])
    if generated_raw is not None:
        native_hw = tuple(clip["frames"]["right"].shape[1:3])
        # The query pool is defined at the native GT resolution.  Upsample the
        # AnyView output before OpenD4RT so the same normalized query locations
        # and aspect token are used for every candidate.
        eval_clip["frames"]["right"] = _resize_video(generated_raw, native_hw)
    target = predicted_target if predicted_target is not None else _critic_target_view(model, eval_clip)
    rng = np.random.default_rng(seed)
    configs = {}
    for config in ("N", "S"):
        stats = _selected_stats(target, eval_clip, source_support, config, angle_thresholds, rng)
        compact = _compact_stats(stats)
        configs[config] = {
            "stats": compact,
            "critic_score_entry": -_combined_error(stats, "mean_entry") if _combined_error(stats, "mean_entry") is not None else None,
            "critic_score_anchor": -_combined_error(stats, "mean_anchor") if _combined_error(stats, "mean_anchor") is not None else None,
            "target_gt_error_entry": _combined_error(stats, "mean_entry"),
            "target_gt_error_anchor": _combined_error(stats, "mean_anchor"),
            "scale": target["fit"],
            "dynamic_d_error_anchor": stats["coverage"]["d"]["dynamic"]["mean_anchor"],
            "static_d_error_anchor": stats["coverage"]["d"]["static"]["mean_anchor"],
        }
    return {"configs": configs}


def _run_anyview(args, episode: Path, out_dir: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "pred.mp4"
    if pred_path.exists() and not args.force:
        return
    command = [
        str(args.anyview_python), str(ROOT / "third_party/AnyView-DVS/scripts/infer.py"),
        "--episode", str(episode), "--input-cam", "cam1", "--target-cam", "cam0",
        "--ckpt", str(args.ckpt), "--tokenizer", str(args.tokenizer),
        "--out", str(out_dir), "--num-frames", str(T_ANYVIEW),
        "--num-steps", str(args.num_steps), "--seed", str(seed),
        "--scale-factor", str(args.scale_factor), "--device", str(args.device),
    ]
    env = os.environ.copy()
    subprocess.run(command, check=True, cwd=str(ROOT), env=env)


def _candidate_rankings(candidates: list[dict], config: str) -> dict:
    names = [candidate["candidate"] for candidate in candidates]
    by_name = {candidate["candidate"]: candidate for candidate in candidates}
    score_order = sorted(
        names,
        key=lambda name: (by_name[name]["score"]["configs"][config]["critic_score_entry"]
                          if by_name[name]["score"]["configs"][config]["critic_score_entry"] is not None
                          else -np.inf),
        reverse=True,
    )
    # The real-right reference has zero RGB error, represented as a null PSNR;
    # place that exact reference first without serializing an infinity.
    psnr_order = sorted(
        names,
        key=lambda name: (np.inf if by_name[name]["rgb"]["psnr_db"] is None
                          else by_name[name]["rgb"]["psnr_db"]),
        reverse=True,
    )
    return {"critic_score_order": score_order, "rgb_psnr_order": psnr_order}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/tmp/kwang-mnt/dynamicreplica/v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenes", default="01f258-3_obj,029beb-3_obj,26b966-3_obj,3592f1-3_obj")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "dynamic_replica_anyview"))
    parser.add_argument("--anyview-python", default=str(ROOT / ".venv-av/bin/python"))
    parser.add_argument("--ckpt", default=str(ROOT / "third_party/AnyView-DVS/checkpoints/anyview_dvs_2b.pt"))
    parser.add_argument("--tokenizer", default=str(ROOT / "third_party/AnyView-DVS/checkpoints/tokenizer.pth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--scale-factor", type=float, default=1.0 / 8.0)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--exp-dir", default=str(ROOT / "third_party/Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG"))
    args = parser.parse_args()

    root = Path(args.data_root)
    out_root = Path(args.out)
    scenes = [scene for scene in args.scenes.split(",") if scene]
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    annotations = _load_annotations(root, args.split)
    missing = sorted(set(scenes) - set(annotations))
    if missing:
        raise ValueError(f"scenes missing from annotations: {missing}")

    model = load_d4rt(args.exp_dir)
    thresholds = dict(LOCKED_THRESHOLDS)
    angle_thresholds = {"e": thresholds["angle_deg_e"], "d": thresholds["angle_deg_d"],
                        "loc": thresholds["angle_deg_loc"]}
    report_clips = []
    for index, scene in enumerate(scenes):
        clip = _load_clip(root, args.split, scene, args.start, args.stride, annotations,
                          args.num_points, 100 + index)
        real_predictions = _run_critic(model, clip)
        source_support = _source_support(
            real_predictions["left"], clip["dynamic"], thresholds["pos_tau"],
            thresholds["loc_tau"], thresholds["disp_abs_tau"], thresholds["disp_ratio"],
            thresholds["min_motion"])
        episode_info = _build_episode(root, args.split, scene, args.start, args.stride,
                                      annotations, out_root / "episodes" / scene,
                                      args.scale_factor)
        frame_ids = episode_info["frame_ids"]
        fg = _load_masks(root, args.split, scene, frame_ids, annotations)
        target_gt = clip["frames"]["right"]
        candidates = []

        real_stats = _score_candidate(model, clip, None, source_support, angle_thresholds, 1000 + index,
                                      predicted_target=real_predictions["right"])
        candidates.append({
            "candidate": "real_right",
            "kind": "real_target_reference",
            "seed": None,
            "rgb": _rgb_diagnostics(target_gt, target_gt, fg),
            "score": real_stats,
        })

        for seed in seeds:
            generated_dir = out_root / "generated" / scene / f"seed_{seed}"
            if not args.skip_generation:
                _run_anyview(args, Path(episode_info["episode"]), generated_dir, seed)
            generated_raw = _read_generated_frames(generated_dir)
            score = _score_candidate(model, clip, generated_raw, source_support,
                                     angle_thresholds, 2000 + index * 100 + seed)
            candidates.append({
                "candidate": f"anyview_seed_{seed}",
                "kind": "anyview_generated",
                "seed": seed,
                "rgb": _rgb_diagnostics(generated_raw, target_gt, fg),
                "score": score,
            })

        report_clips.append({
            "scene": scene,
            "split": args.split,
            "frame_ids": frame_ids,
            "episode": episode_info,
            "sampling": clip["sampling"],
            "source_support": _support_summary(source_support, clip["dynamic"]),
            "source_support_protocol": "fixed from real left video before any candidate generation; N/S share it",
            "candidate_pool": "same scene query points, target GT visibility, OpenD4RT critic, and per-candidate full-V scale fit",
            "candidates": candidates,
            "rankings": {config: _candidate_rankings(candidates, config) for config in ("N", "S")},
            "failure_mode_proxies": {
                "deletion": "foreground_l1 and foreground PSNR against real right RGB; high error with a high D4RT score is a warning",
                "motion_loss": "dynamic-anchor d error and foreground temporal-motion L1",
                "camera_follow": "static-anchor d error and background temporal-motion L1",
            },
        })
        print(json.dumps({"scene": scene, "sampled_dynamic": clip["sampling"]["sampled_dynamic"],
                          "sampled_static": clip["sampling"]["sampled_static"],
                          "generated_candidates": len(seeds)}, ensure_ascii=False), flush=True)

    result = {
        "experiment": "dynamic_replica_anyview_real_stereo_validation",
        "protocol": {
            "no_formal_training": True,
            "generation_frames": T_ANYVIEW,
            "critic_and_geometry_frames": T_D4RT,
            "critic_frames": "first 40 frames of the 41-frame AnyView output, resized back to native target resolution before querying",
            "primary_configs": ["N", "S"],
            "angle_gate": "not used; SV remains analysis-only in the preceding calibration",
            "source_masks": "generated once from real left predictions and locked thresholds, then shared by every candidate",
            "scale": "one fixed-V scale fit per candidate, reused by N/S and all term subsets",
            "real_right_role": "target-GT is used for the fixed-V scale and target-error evaluation; right RGB is an auxiliary diagnostic; neither enters source masks",
            "anyview_scale_factor": args.scale_factor,
        },
        "data": {"root": str(root), "split": args.split, "scenes": scenes,
                 "start": args.start, "stride": args.stride, "num_points": args.num_points,
                 "seeds": seeds},
        "thresholds": thresholds,
        "clips": report_clips,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "report.json"
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "clips": len(report_clips)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
