"""Length-aware allocation of the decoding budget across view batches.

Threshold DFS needs at least ``L`` sequential forward passes before it can
record any candidate of ``L`` answer tokens.  Splitting a fixed budget equally
over all view batches can make every slice shorter than ``L * step_seconds``;
then no view finishes even its greedy path and the candidate pool stays empty.

This module predicts the answer length from the demonstrations, then chooses
how many view batches to decode for each test output so that every decoded
slice can at least complete one greedy path.  It changes only the time
allocation; the decoder, scores, and selection are unchanged.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence


def answer_tokens(height: int, width: int) -> int:
    """Tokens of a reply: one per cell, ``height-1`` newlines, one end token.

    ``QwenFormatter.fmt_reply`` writes rows joined by newlines followed by the
    end marker; the decoder records the end marker as its final token.
    """
    return height * width + height


def _shape(grid) -> tuple:
    return len(grid), len(grid[0])


def predict_output_shape(train: Sequence[dict], test_input) -> tuple:
    """Return ``(shape, rule)`` for the test output, with ``shape=None`` if unknown.

    Rules are tried in order and must hold on every demonstration:
    identity shape, constant output shape, constant rational scale, constant
    additive offset.
    """
    pairs = [(_shape(p["input"]), _shape(p["output"])) for p in train]
    th, tw = _shape(test_input)
    if all(i == o for i, o in pairs):
        return (th, tw), "same_as_input"
    outs = {o for _, o in pairs}
    if len(outs) == 1:
        return next(iter(outs)), "constant_output"
    ratios = {(Fraction(o[0], i[0]), Fraction(o[1], i[1])) for i, o in pairs}
    if len(ratios) == 1:
        rh, rw = next(iter(ratios))
        h, w = rh * th, rw * tw
        if h.denominator == 1 and w.denominator == 1 and 1 <= h <= 30 and 1 <= w <= 30:
            return (int(h), int(w)), "constant_scale"
    offsets = {(o[0] - i[0], o[1] - i[1]) for i, o in pairs}
    if len(offsets) == 1:
        dh, dw = next(iter(offsets))
        h, w = th + dh, tw + dw
        if 1 <= h <= 30 and 1 <= w <= 30:
            return (h, w), "constant_offset"
    return None, "unknown"


def estimate_answer_tokens(train: Sequence[dict], test_input) -> tuple:
    """Return ``(tokens, rule)``; unknown shapes use the largest plausible size.

    The fallback takes the maximum of the demonstration outputs and the test
    input in each dimension, which over-estimates rather than under-estimates
    the time a greedy path needs.
    """
    shape, rule = predict_output_shape(train, test_input)
    if shape is None:
        dims = [_shape(p["output"]) for p in train] + [_shape(test_input)]
        shape = (max(d[0] for d in dims), max(d[1] for d in dims))
    return answer_tokens(*shape), rule


def plan_view_batches(available: int, budget_seconds: float, est_tokens: int,
                      step_seconds: float, margin: float = 1.15,
                      prefill_seconds: float = 2.0) -> int:
    """Number of view batches to decode for one test output.

    Uses as many batches as the budget allows while keeping each slice at least
    ``margin`` times the estimated greedy completion time.  Always decodes at
    least one batch and never more than ``available``.
    """
    if available < 1:
        return 0
    need = margin * est_tokens * step_seconds + prefill_seconds
    k = int(budget_seconds // need) if need > 0 else available
    return max(1, min(available, k))


def allocate_task(outputs: Sequence[dict], budget_seconds: float, step_seconds: float,
                  batches_per_output: int = 4, margin: float = 1.15,
                  prefill_seconds: float = 2.0) -> list:
    """Plan every test output of a task under one shared budget.

    ``outputs`` items need ``est_tokens``.  The budget is split equally across
    outputs, as in the parent loop.  Returns a list of dicts with the number of
    batches and the per-batch slice.
    """
    if not outputs:
        return []
    per_output = budget_seconds / len(outputs)
    plan = []
    for item in outputs:
        k = plan_view_batches(batches_per_output, per_output, item["est_tokens"], step_seconds,
                              margin, prefill_seconds)
        plan.append({"batches": k, "slice_seconds": per_output / k})
    return plan


def greedy_reachable(tokens: int, slice_seconds: float, step_seconds: float,
                     prefill_seconds: float = 0.0) -> bool:
    """Whether a slice can complete at least one greedy path of ``tokens``."""
    return tokens * step_seconds + prefill_seconds <= slice_seconds


def select_batches(batch_list: Sequence, k: int) -> list:
    """Keep ``k`` batches while spreading them across geometric groups.

    The parent orders the four batches of an output as identity/rot180,
    rot90/rot270, transpose, transpose-rotations.  Taking every
    ``len/k``-th batch keeps one non-transposed and one transposed group when
    ``k=2``.
    """
    n = len(batch_list)
    if k >= n:
        return list(batch_list)
    step = n / k
    return [batch_list[int(i * step)] for i in range(k)]

