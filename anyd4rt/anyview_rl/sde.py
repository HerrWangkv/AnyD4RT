"""Flow-GRPO style stochastic transition for AnyView, in rectified-flow z coordinates (README §4.1-4.4).

AnyView state is VE: x = x0 + sigma * eps. With tau = sigma / (1 + sigma) and z = x / (1 + sigma):
    z = (1 - tau) x0 + tau eps,   v = eps - x0 = (z - x0) / tau.
Only the math of one transition lives here; the network call (x0 prediction) is the caller's.

Precision: log_prob / kl compute in `dtype` (default float32, README §4.2); tests pass float64.
Batch: tensors are [B, ...]; log_prob / kl return one value per candidate ([B]), averaged over the
non-batch dimensions selected by `mask` (e.g. the rgb0 target region only, README §4.1).
"""

from __future__ import annotations

import math

import torch


def tau_of(sigma):
    return sigma / (1.0 + sigma)


def x_to_z(x, sigma):
    return x / (1.0 + sigma)


def z_to_x(z, sigma):
    return z * (1.0 + sigma)


def velocity(z, x0_hat, tau):
    """v_hat = eps_hat - x0_hat, with eps_hat implied by z and x0_hat."""
    return (z - x0_hat) / tau


def transition_mean_std(z, x0_hat, tau, tau_next, a: float):
    """Mean and std of p(z_next | z) (README §4.2). tau decreases: delta = tau_next - tau < 0.

    mu  = z + [v + g^2 / (2 tau) (z + (1 - tau) v)] * delta,   g = a * sqrt(tau / (1 - tau))
    std = g * sqrt(-delta)   (a python float; tau, tau_next are scalars)
    a = 0 gives the deterministic Euler step (std = 0).
    """
    tau, tau_next = float(tau), float(tau_next)
    delta = tau_next - tau
    v = velocity(z, x0_hat, tau)
    g2 = (a * a) * tau / (1.0 - tau)
    mu = z + (v + g2 / (2.0 * tau) * (z + (1.0 - tau) * v)) * delta
    return mu, math.sqrt(g2 * (-delta))


def step(z, x0_hat, tau, tau_next, a: float, noise=None):
    """One transition. noise ~ N(0, I) of z's shape (required when a > 0). Returns (z_next, mu, std)."""
    mu, std = transition_mean_std(z, x0_hat, tau, tau_next, a)
    if a == 0.0:
        return mu, mu, std
    return mu + std * noise, mu, std


def _batch_mean(t, mask):
    """Mean over all non-batch dims ([B, ...] -> [B]); with mask ([B or 1, ...] bool), only where mask is True."""
    dims = tuple(range(1, t.ndim))
    if mask is None:
        return t.mean(dim=dims)
    m = mask.to(t.dtype).expand_as(t)
    return (t * m).sum(dim=dims) / m.sum(dim=dims)


def log_prob(z_next, mu, std: float, mask=None, dtype=torch.float32):
    """Per-candidate, per-dimension mean Gaussian log-density ([B]). Undefined for std = 0 (a = 0)."""
    if not (std > 0):
        raise ValueError("log-prob undefined for a zero-variance transition (a = 0)")
    z_next, mu = z_next.to(dtype), mu.to(dtype)
    ll = -((z_next - mu) ** 2) / (2.0 * std**2) - math.log(std) - 0.5 * math.log(2.0 * math.pi)
    return _batch_mean(ll, mask)


def kl(mu_theta, mu_ref, std: float, mask=None, dtype=torch.float32):
    """Exact KL between equal-variance Gaussians, per candidate, per-dimension mean ([B], README §4.4)."""
    return _batch_mean(((mu_theta.to(dtype) - mu_ref.to(dtype)) ** 2) / (2.0 * std**2), mask)
