"""Our own sampling, so the generation loop is ours.

Each filter is a small pure function over a ``(batch, vocab)`` logits tensor, which
makes them testable without a model and keeps us independent of whichever
``LogitsProcessor`` names the current transformers major version happens to use.

Order matters and follows the usual convention: repetition penalty on raw logits,
then temperature, then top-k, then top-p, then min-p.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

NEG_INF = float("-inf")


def apply_repetition_penalty(
    logits: torch.Tensor, generated: Sequence[int] | torch.Tensor, penalty: float
) -> torch.Tensor:
    """Divide the logits of already-generated tokens (multiply if negative).

    The CTRL formulation: dividing a positive logit and multiplying a negative one both
    push the token down, which a plain subtraction would not.
    """
    if penalty == 1.0 or len(generated) == 0:
        return logits
    ids = torch.as_tensor(list(generated), device=logits.device, dtype=torch.long).unique()
    selected = logits[..., ids]
    logits[..., ids] = torch.where(selected > 0, selected / penalty, selected * penalty)
    return logits


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0 or temperature == 1.0:
        return logits
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, NEG_INF)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose mass reaches ``p``."""
    if p >= 1.0 or p <= 0.0:
        return logits
    ordered, indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
    # Drop everything after the cumulative mass passes p, but never the top token.
    remove = cumulative - torch.softmax(ordered, dim=-1) >= p
    remove[..., 0] = False
    return logits.masked_fill(remove.scatter(-1, indices, remove), NEG_INF)


def min_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep tokens at least ``p`` times as likely as the most likely one."""
    if p <= 0.0:
        return logits
    probabilities = torch.softmax(logits, dim=-1)
    threshold = probabilities.max(dim=-1, keepdim=True).values * p
    return logits.masked_fill(probabilities < threshold, NEG_INF)


def filter_logits(
    logits: torch.Tensor,
    *,
    generated: Sequence[int] | torch.Tensor = (),
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    logits = logits.clone().float()
    logits = apply_repetition_penalty(logits, generated, repetition_penalty)
    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    logits = min_p_filter(logits, min_p)
    return logits


def choose(
    logits: torch.Tensor,
    *,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick one token per row. Returns a ``(batch,)`` long tensor."""
    if greedy:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)


def sample_next(
    logits: torch.Tensor,
    params,
    *,
    generated: Sequence[int] | torch.Tensor = (),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One decode step's worth of sampling, driven by ``GenerationParams``."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if params.greedy:
        # Only the repetition penalty matters when we are taking the argmax.
        filtered = apply_repetition_penalty(
            logits.clone().float(), generated, params.repetition_penalty
        )
        return choose(filtered, greedy=True)
    filtered = filter_logits(
        logits,
        generated=generated,
        temperature=params.temperature,
        top_k=params.top_k,
        top_p=params.top_p,
        min_p=params.min_p,
        repetition_penalty=params.repetition_penalty,
    )
    return choose(filtered, greedy=False, generator=generator)
