"""AnyView sampling with a replaceable per-step transition (README §4.2, §6 S3 / B4).

Reuses the released pipeline unchanged: stream packing, starting noise, sigma schedule, the network
call `pipe.denoise` (preconditioning + masked assembly of conditioning inputs), and the final clean
pass at sigma_min. Only the state update between network calls is replaced:
  mode="ab2":   the released RectifiedFlowAB2Scheduler step (parity check with pipe.generate)
  mode="euler": deterministic Euler in z coordinates on every step (= sde with a = 0)
  mode="sde":   Flow-GRPO transition with noise level a on every step
Noise is added only where masks['output'] is set (the rgb0 target channels of stream v0); all other
channels are overwritten by the conditioning inputs inside pipe.denoise at every step.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

from anyd4rt.anyview_rl import sde

AV = Path(__file__).resolve().parents[2] / "third_party" / "AnyView-DVS"
if str(AV) not in sys.path:
    sys.path.insert(0, str(AV))

from anyview.logistics import pack_streams_from_entries  # noqa: E402
from anyview.pipe import scheduler_step_streams  # noqa: E402
from anyview.vendor.misc import arch_invariant_rand  # noqa: E402


@dataclass
class Rollout:
    y0_pred_streams: dict  # final prediction, same format as pipe.generate()["y0_pred_streams"]
    # Per stochastic step (mode="sde", record=True): the float32 state x_i fed to the network, z_{i+1} (float64),
    # std and the old log-prob, for the stream that has output channels.
    steps: list = field(default_factory=list)


def _setup(pipe, entries, seed, num_steps):
    x0_streams, masks = pack_streams_from_entries(entries)
    assert list(x0_streams.keys()) == ["v0", "v1"]
    B = x0_streams["v0"].shape[0]
    shapes = {k: v.shape[1:] for k, v in x0_streams.items()}
    crossattn = pipe.empty_text_emb.repeat(B, 1, 1)
    start = {k: arch_invariant_rand((B,) + tuple(shapes[k]), torch.float32, pipe.tensor_kwargs["device"], seed * 100 + v)
             * pipe.scheduler.config.sigma_max for v, k in enumerate(shapes)}
    pipe.scheduler.set_timesteps(num_steps, device=x0_streams["v0"].device)
    return x0_streams, masks, crossattn, start


def sigma_in(pipe, i, sample):
    s = pipe.scheduler.sigmas[i].to(sample["v0"].device, dtype=torch.float32)
    return {k: s.repeat(v.shape[0], *([1] * (v.ndim - 1))) for k, v in sample.items()}


@torch.no_grad()
def sample(pipe, entries, seed=0, num_steps=35, mode="ab2", a=0.0, noise_seed=0, record=False) -> Rollout:
    x0_streams, masks, crossattn, state = _setup(pipe, entries, seed, num_steps)
    sig = pipe.scheduler.sigmas.double()
    gen = torch.Generator(device=state["v0"].device).manual_seed(noise_seed)
    out_mask = {k: masks["output"][k].bool() for k in state}
    y0_prev, steps = None, []
    for i in range(num_steps):
        _, y0_pred = pipe.denoise(x0_streams, state, masks, sigma_in(pipe, i, state), crossattn)
        if mode == "ab2":
            state, y0_prev = scheduler_step_streams(pipe.scheduler, x0_pred_streams=y0_pred, i=i,
                                                    sample_streams=state, x0_prev_streams=y0_prev)
            continue
        s_t, s_s = float(sig[i]), float(sig[i + 1])
        t_t, t_s = sde.tau_of(s_t), sde.tau_of(s_s)
        new = {}
        for k, x in state.items():
            z = sde.x_to_z(x.double(), s_t)
            noise = None
            if mode == "sde":
                noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=torch.float64) * out_mask[k]
            z_next, mu, std = sde.step(z, y0_pred[k].double(), t_t, t_s, a if mode == "sde" else 0.0, noise)
            new[k] = sde.z_to_x(z_next, s_s).to(torch.float32)
            if record and mode == "sde" and out_mask[k].any():
                steps.append({"i": i, "stream": k, "x": x.clone(), "z_next": z_next.clone(), "std": std,
                              "logp_old": sde.log_prob(z_next, mu, std, mask=out_mask[k])})
        state = new
    _, final = pipe.denoise(x0_streams, state, masks, sigma_in(pipe, num_steps, state), crossattn)
    from anyview.logistics import assemble_clean_predictions
    return Rollout(assemble_clean_predictions(x0_streams, final, masks), steps)


@torch.no_grad()
def recompute_log_prob(pipe, entries, step_rec, num_steps=35, a=0.7, other_state=None):
    """Recompute log p(z_{i+1} | z_i) with a fresh network call (README §4.5 policy recomputation).

    Only the output region of the recorded stream matters for the network input; `other_state`
    optionally fills every other stream / channel (e.g. with random values) to check that.
    """
    x0_streams, masks, crossattn, start = _setup(pipe, entries, 0, num_steps)
    i, k = step_rec["i"], step_rec["stream"]
    sig = pipe.scheduler.sigmas.double()
    s_t, s_s = float(sig[i]), float(sig[i + 1])
    state = other_state if other_state is not None else {kk: torch.zeros_like(v) for kk, v in start.items()}
    state = dict(state)
    om = masks["output"][k].bool()
    state[k] = torch.where(om, step_rec["x"], state[k])  # the exact float32 state the rollout fed to the network
    _, y0 = pipe.denoise(x0_streams, state, masks, sigma_in(pipe, i, state), crossattn)
    mu, std = sde.transition_mean_std(sde.x_to_z(state[k].double(), s_t), y0[k].double(), sde.tau_of(s_t), sde.tau_of(s_s), a)
    return sde.log_prob(step_rec["z_next"], mu, std, mask=om), mu
