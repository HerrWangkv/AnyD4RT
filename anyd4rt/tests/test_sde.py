"""Transition math only (README §6 S3). No network: x0 predictions are random tensors.

What these tests do NOT cover: noise scaling inside the real sampler, log-prob recomputation
through the network, bf16 effects, and the conditioning channels (v1 stream, cams) handling.
"""

import sys
from pathlib import Path

import pytest
import torch

from anyd4rt.anyview_rl import sde

AV = Path(__file__).resolve().parents[2] / "third_party" / "AnyView-DVS"
sys.path.insert(0, str(AV))
from anyview.vendor.rectified_flow_scheduler import RectifiedFlowAB2Scheduler  # noqa: E402
from anyview.vendor.runge_kutta import reg_x0_euler_step  # noqa: E402

SHAPE = (2, 16, 3, 6, 9)


def _sigmas(n):
    s = RectifiedFlowAB2Scheduler()
    s.set_timesteps(n)
    return s.sigmas.double()


@pytest.mark.parametrize("n_steps", [35, 16, 12, 10])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a0_matches_released_euler_on_every_step(n_steps, seed):
    """a = 0 in z coordinates == AnyView's x0-form Euler step in x coordinates, for every sigma pair."""
    sig = _sigmas(n_steps)
    g = torch.Generator().manual_seed(seed)
    for i in range(n_steps):
        s_t, s_s = sig[i], sig[i + 1]
        x0_hat = torch.randn(SHAPE, generator=g, dtype=torch.float64)
        x_t = x0_hat + s_t * torch.randn(SHAPE, generator=g, dtype=torch.float64) + 0.3 * torch.randn(SHAPE, generator=g, dtype=torch.float64)
        ones = torch.ones(SHAPE[0], dtype=torch.float64)
        ref, _ = reg_x0_euler_step(x_t, s_t * ones, s_s * ones, x0_hat)
        z_next, _, std = sde.step(sde.x_to_z(x_t, s_t), x0_hat, sde.tau_of(s_t), sde.tau_of(s_s), a=0.0)
        assert std == 0
        err = (sde.z_to_x(z_next, s_s) - ref).abs().max() / (ref.abs().max() + 1e-12)
        assert err < 1e-12, (n_steps, i, float(s_t), float(err))


def test_standardized_residual_has_unit_scale():
    sig = _sigmas(35)
    g = torch.Generator().manual_seed(0)
    for i in (0, 10, 25, 34):
        tau, tau_n = sde.tau_of(sig[i]), sde.tau_of(sig[i + 1])
        z = torch.randn(200_000, generator=g, dtype=torch.float64)
        x0 = torch.randn(200_000, generator=g, dtype=torch.float64)
        noise = torch.randn(200_000, generator=g, dtype=torch.float64)
        z_next, mu, std = sde.step(z, x0, tau, tau_n, a=0.7, noise=noise)
        r = (z_next - mu) / std
        assert abs(float(r.mean())) < 0.01 and abs(float(r.var()) - 1) < 0.02


def test_log_prob_and_kl_identities():
    sig = _sigmas(35)
    tau, tau_n = sde.tau_of(sig[5]), sde.tau_of(sig[6])
    g = torch.Generator().manual_seed(0)
    z, x0, noise = (torch.randn(1000, generator=g, dtype=torch.float64) for _ in range(3))
    z_next, mu, std = sde.step(z, x0, tau, tau_n, a=0.7, noise=noise)
    lp = sde.log_prob(z_next[None], mu[None], std, dtype=torch.float64)
    lp_ref = torch.distributions.Normal(mu, std).log_prob(z_next).mean()
    assert lp.shape == (1,) and torch.allclose(lp[0], lp_ref, atol=1e-12)
    assert float(sde.kl(mu[None], mu[None], std)[0]) == 0.0
    with pytest.raises(ValueError):
        sde.log_prob(z_next[None], mu[None], 0.0)


def test_log_prob_and_kl_keep_batch_dim_and_mask():
    """GRPO needs one value per candidate; the mask restricts to the target (rgb0) region."""
    g = torch.Generator().manual_seed(0)
    B = 4
    mu = torch.randn(B, 3, 5, 7, generator=g, dtype=torch.float64)
    z_next = mu + 0.3 * torch.randn(B, 3, 5, 7, generator=g, dtype=torch.float64)
    z_next[1] += 5.0  # only candidate 1 is off
    lp = sde.log_prob(z_next, mu, 0.3, dtype=torch.float64)
    assert lp.shape == (B,) and int(torch.argmin(lp)) == 1
    per = [float(sde.log_prob(z_next[b:b + 1], mu[b:b + 1], 0.3, dtype=torch.float64)[0]) for b in range(B)]
    assert torch.allclose(lp, torch.tensor(per, dtype=torch.float64))
    mask = torch.zeros(1, 3, 5, 7, dtype=torch.bool)
    mask[:, 0] = True  # e.g. first channel group only
    lpm = sde.log_prob(z_next, mu, 0.3, mask=mask, dtype=torch.float64)
    ref = sde.log_prob(z_next[:, :1], mu[:, :1], 0.3, dtype=torch.float64)
    assert torch.allclose(lpm, ref)
    k = sde.kl(mu + 0.1, mu, 0.3, mask=mask)
    assert k.shape == (B,) and k.dtype == torch.float32 and torch.allclose(k, torch.full((B,), 0.01 / 0.18))


def test_default_precision_is_fp32():
    mu = torch.zeros(2, 8, dtype=torch.bfloat16)
    assert sde.log_prob(mu, mu, 0.5).dtype == torch.float32


def test_noise_scale_magnitudes_match_readme():
    """README §4.2 (measured): a = 0.7, 35 steps: per-step noise 0.285-0.308 for sigma > 10; the first step with
    sigma < 0.1 is 0.033, all later ones < 0.03; the drift correction coefficient g^2(-delta)/(2 tau) <= 0.0523."""
    sig = _sigmas(35)
    stds = []
    for i in range(35):
        _, std = sde.transition_mean_std(torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
                                         sde.tau_of(sig[i]), sde.tau_of(sig[i + 1]), a=0.7)
        stds.append(float(std))
    hi = [s for s, sg in zip(stds, sig[:-1]) if sg > 10]
    lo = [s for s, sg in zip(stds, sig[:-1]) if sg < 0.1]
    assert 0.285 < min(hi) and max(hi) < 0.308
    assert 0.032 < lo[0] < 0.034 and max(lo[1:]) < 0.03
    corr = [0.49 * float(sde.tau_of(sig[i])) / (1 - float(sde.tau_of(sig[i]))) * (float(sde.tau_of(sig[i])) - float(sde.tau_of(sig[i + 1])))
            / (2 * float(sde.tau_of(sig[i]))) for i in range(35)]
    assert max(corr) < 0.0523 + 1e-4
