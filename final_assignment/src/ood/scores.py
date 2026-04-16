from __future__ import annotations

import torch


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


@torch.no_grad()
def pixel_max_softmax(logits: torch.Tensor) -> torch.Tensor:
    probs = softmax_probs(logits)
    return probs.max(dim=1).values


@torch.no_grad()
def pixel_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = softmax_probs(logits)
    log_probs = torch.log(probs.clamp_min(eps))
    return -(probs * log_probs).sum(dim=1)


@torch.no_grad()
def pixel_energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    # Energy-based OOD score from logits.
    # Higher energy can be treated as more OOD-like after sign convention below.
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


@torch.no_grad()
def image_mean_msp(logits: torch.Tensor) -> torch.Tensor:
    return pixel_max_softmax(logits).mean(dim=(1, 2))


@torch.no_grad()
def image_mean_entropy(logits: torch.Tensor) -> torch.Tensor:
    return pixel_entropy(logits).mean(dim=(1, 2))


@torch.no_grad()
def image_mean_energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return pixel_energy(logits, temperature=temperature).mean(dim=(1, 2))


@torch.no_grad()
def image_percentile_entropy(
    logits: torch.Tensor,
    percentile: float = 95.0,
) -> torch.Tensor:
    ent = pixel_entropy(logits)
    batch_size = ent.shape[0]
    flat = ent.view(batch_size, -1)
    return torch.quantile(flat, q=percentile / 100.0, dim=1)


@torch.no_grad()
def image_percentile_low_msp(
    logits: torch.Tensor,
    percentile: float = 5.0,
) -> torch.Tensor:
    msp = pixel_max_softmax(logits)
    batch_size = msp.shape[0]
    flat = msp.view(batch_size, -1)
    return torch.quantile(flat, q=percentile / 100.0, dim=1)