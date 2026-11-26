"""Flow matching as Wan2.1 uses it: noising, the training target, how noise levels are drawn, and a sampler.

    x_sigma = (1 - sigma) * x0 + sigma * noise          sigma in [0, 1], time step t = 1000 * sigma
    the network predicts   noise - x0

Training follows DiffSynth-Studio's Wan trainer (``FlowMatchScheduler(shift=5).set_timesteps(1000,
training=True)``, v1.1.7): one of 1000 levels is drawn uniformly, sigma = shift * u / (1 + (shift - 1) * u)
with u linearly spaced from 1 down to 0, and the loss of a sample is weighted by a bell over the time step
that is normalised to mean 1.

Sampling is the plain Euler rule on the same shifted levels (Wan's own default is UniPC with shift 5 and
guidance 5 at 50 steps; Euler needs no extra state and is enough for a first look).
"""
from __future__ import annotations

import torch

NUM_TRAIN_STEPS = 1000


def shifted(u: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * u / (1 + (shift - 1) * u)


def training_levels(shift: float = 5.0, n: int = NUM_TRAIN_STEPS):
    """(sigmas (n,), weights (n,)): the discrete noise levels of training and the loss weight of each."""
    u = torch.linspace(1.0, 0.0, n + 1, dtype=torch.float64)[:-1]
    sigmas = shifted(u, shift)
    t = sigmas * n
    y = torch.exp(-2 * ((t - n / 2) / n) ** 2)
    y = y - y.min()
    weights = y * (n / y.sum())
    return sigmas.float(), weights.float()


def draw_levels(batch: int, shift: float = 5.0, weighting: str = "bell", generator=None, device=None):
    """Noise level and loss weight of every sample: (sigma (B,), weight (B,))."""
    sigmas, weights = training_levels(shift)
    idx = torch.randint(0, sigmas.numel(), (batch,), generator=generator)
    w = weights[idx] if weighting == "bell" else torch.ones(batch)
    return sigmas[idx].to(device), w.to(device)


def add_noise(x0: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    s = sigma.view(-1, *([1] * (x0.dim() - 1))).to(x0.dtype)
    return (1 - s) * x0 + s * noise


def velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return noise - x0


def sampling_levels(steps: int, shift: float = 5.0) -> torch.Tensor:
    """steps + 1 noise levels from 1 down to 0."""
    return shifted(torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64), shift).float()


@torch.no_grad()
def sample_euler(predict, shape, steps: int = 50, shift: float = 5.0, generator=None, device=None,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
    """Integrate from pure noise to a sample. ``predict(x, t)`` returns the network output (noise - x0) for
    latents x and time steps t (B,) in 0..1000 (guidance, if any, is the caller's business)."""
    if noise is None:
        noise = torch.randn(shape, generator=generator, device=generator.device if generator is not None else "cpu")
    x = noise.to(device)
    sigmas = sampling_levels(steps, shift).to(x.device)
    for i in range(steps):
        t = torch.full((x.shape[0],), float(sigmas[i]) * NUM_TRAIN_STEPS, device=x.device)
        v = predict(x, t)
        x = x + (sigmas[i + 1] - sigmas[i]) * v.to(x.dtype)
    return x
