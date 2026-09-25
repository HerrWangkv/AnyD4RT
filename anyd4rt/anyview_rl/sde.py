"""Flow-GRPO style stochastic transition for AnyView, in rectified-flow z coordinates (README §4.1-4.4).

AnyView state is VE: x = x0 + sigma * eps. With tau = sigma / (1 + sigma) and z = x / (1 + sigma):
    z = (1 - tau) x0 + tau eps,   v = eps - x0 = (z - x0) / tau.
Only the math of one transition lives here; the network call (x0 prediction) is the caller's.
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
    std = g * sqrt(-delta)
    a = 0 gives the deterministic Euler step (std = 0).
    """
    delta = tau_next - tau
    v = velocity(z, x0_hat, tau)
    g2 = (a * a) * tau / (1.0 - tau)
    mu = z + (v + g2 / (2.0 * tau) * (z + (1.0 - tau) * v)) * delta
    std = math.sqrt(g2 * (-delta)) if not torch.is_tensor(tau) else torch.sqrt(g2 * (-delta))
    return mu, std


def step(z, x0_hat, tau, tau_next, a: float, noise=None):
    """One transition. noise ~ N(0, I) of z's shape (required when a > 0)."""
    mu, std = transition_mean_std(z, x0_hat, tau, tau_next, a)
    if a == 0.0:
        return mu, mu, std
    return mu + std * noise, mu, std


def log_prob_mean(z_next, mu, std):
    """Per-dimension mean Gaussian log-density, in float32 or wider (README §4.3). Not defined for std = 0."""
    if not (std > 0):
        raise ValueError("log-prob undefined for a zero-variance transition (a = 0)")
    z_next, mu = z_next.double(), mu.double()
    ll = -((z_next - mu) ** 2) / (2.0 * std**2) - math.log(std) - 0.5 * math.log(2.0 * math.pi)
    return ll.mean()


def kl_mean(mu_theta, mu_ref, std):
    """Exact KL between equal-variance Gaussians, per-dimension mean (README §4.4)."""
    return (((mu_theta.double() - mu_ref.double()) ** 2) / (2.0 * std**2)).mean()
