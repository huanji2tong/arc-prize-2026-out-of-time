"""Iterative threshold DFS that reproduces the parent ``turbo_dfs`` search.

The parent Notebook (``notebooks/baseline_33_89_source.py``, lines 506-595)
explores continuations recursively: one Python frame per generated token, a
full-vocabulary ``log_softmax`` copied to the CPU at every step, and one
``.item()`` call per ARC token and row.  This module keeps the same search
semantics and changes only how the work is executed:

* an explicit stack replaces recursion, so answer length is not bounded by the
  interpreter recursion limit (a 30x30 grid needs 931 decoding levels);
* the log-softmax is computed where the logits live and only the ARC-token
  columns (16 of ~152k) are transferred to the host;
* the per-row candidate lists are built from one ``tolist()`` call.

Search semantics that are preserved exactly (see ``scripts/test_fast_dfs.py``):

* rows of a batch advance in lockstep; at every level each row pops its next
  candidate, exhausted rows are fed ``pad_id`` with score 1000;
* children are visited in stable ascending score order, ties keep the order
  of ``arc_tokens``;
* an end token within the cutoff is recorded as a finished suffix at the level
  where it appears; non-end tokens are only expanded while
  ``max_new_tokens > 1``;
* the deadline test ``now - start < time_cap and now < end_time`` is applied
  at the top of every level loop, so a deadline unwinds the stack in the same
  order as the recursive version;
* the cache returned by the forward call at depth ``d`` is the cache passed to
  every child of depth ``d``.  With the Unsloth inference path the length of
  the passed cache fixes the write position, which is what makes sibling
  branches overwrite each other correctly.  ``prepare_cache`` lets a caller
  enforce the same semantics on an in-place cache (for example by cropping a
  ``transformers`` ``DynamicCache`` to ``pos``).

When ``device_log_softmax`` is true the log-softmax is evaluated on the model
device.  On CUDA the reduction order differs from the CPU reduction used by the
parent, so scores may differ in the last float32 bits; set it to false to keep
the parent arithmetic bit-for-bit.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Optional

import torch

DEAD_SCORE = 1000.0


class _Frame:
    __slots__ = ("pos", "cache", "remaining", "candidates", "suffixes", "child_tokens")

    def __init__(self, pos, cache, remaining, candidates, suffixes):
        self.pos = pos
        self.cache = cache
        self.remaining = remaining          # max_new_tokens at this level
        self.candidates = candidates        # list[list[(score, token)]] per row
        self.suffixes = suffixes            # defaultdict(list) per row
        self.child_tokens = None            # tokens fed to the active child


def _arc_log_probs(logits: torch.Tensor, arc_index: torch.Tensor, device_log_softmax: bool) -> torch.Tensor:
    """Return float32 CPU log probabilities of the ARC-token columns, shape (n, |arc|)."""
    if device_log_softmax:
        logp = logits.float().log_softmax(-1).index_select(-1, arc_index.to(logits.device))
        return logp.cpu()
    return logits.float().cpu().log_softmax(-1).index_select(-1, arc_index)


def _expand(logits, scores, remaining, max_score, arc_tokens, arc_index, eos_id,
            device_log_softmax, trace):
    """Build the candidate lists and finished suffixes of one level."""
    if trace is not None:
        trace["prefix_calls"] = trace.get("prefix_calls", 0) + 1
    logp = _arc_log_probs(logits, arc_index, device_log_softmax)
    n = logp.size(0)
    # Same float32 operation as the parent: tensor(scores).view(n, 1) - log_softmax.
    rows = (torch.tensor(scores, dtype=torch.float32).view(n, 1) - logp).tolist()
    candidates, suffixes = [], defaultdict(list)
    for i, row in enumerate(rows):
        found = []
        for t, score in zip(arc_tokens, row):
            if score < max_score:
                if t == eos_id:
                    if trace is not None:
                        trace["eos_within_cutoff"] = trace.get("eos_within_cutoff", 0) + 1
                    suffixes[i].append((score, [t]))
                elif remaining > 1:
                    found.append((score, t))
            elif trace is not None:
                trace["pruned_extensions"] = trace.get("pruned_extensions", 0) + 1
        found.sort(key=lambda x: x[0])
        candidates.append(found)
    return candidates, suffixes


def _merge_into_parent(child: _Frame, parent: _Frame) -> None:
    for batch_id, beams in child.suffixes.items():
        head = parent.child_tokens[batch_id]
        for score, tokens in beams:
            parent.suffixes[batch_id].append((score, [head] + tokens))


def turbo_dfs_iterative(model, logits, max_new_tokens, max_score, scores, pos, cache,
                        start_time, end_time, *, arc_tokens, eos_id, pad_id,
                        time_cap: float = 540.0, device_log_softmax: bool = True,
                        prepare_cache: Optional[Callable] = None, trace: Optional[dict] = None,
                        clock: Callable[[], float] = time.time,
                        extend_until: Optional[float] = None) -> dict:
    """Drop-in iterative replacement for the parent ``turbo_dfs``.

    Returns ``{row: [(score, tokens), ...]}`` with the same content and append
    order as the recursive version.

    ``extend_until`` (default off) enables first-descent extension: while the
    batch is still on its first greedy descent and no row has finished a
    candidate, the deadline is ``extend_until`` instead of ``end_time``.  The
    extension ends as soon as any suffix is recorded, any level starts its
    second child, or every row has been pruned.  With ``extend_until=None``
    the search is identical to the parent.
    """
    arc_tokens = list(arc_tokens)
    arc_index = torch.tensor(arc_tokens, dtype=torch.long)
    n = logits.size(0)

    def expand(level_logits, level_scores, remaining):
        return _expand(level_logits, level_scores, remaining, max_score, arc_tokens, arc_index,
                       eos_id, device_log_softmax, trace)

    cands, sufs = expand(logits, list(scores), max_new_tokens)
    root = _Frame(pos, cache, max_new_tokens, cands, sufs)
    stack = [root]
    device = getattr(model, "device", logits.device)
    extending = extend_until is not None and not sufs

    while stack:
        frame = stack[-1]
        deadline = extend_until if extending else end_time
        # Two clock reads with short-circuit, exactly as in the parent loop condition.
        if not (clock() - start_time < time_cap and clock() < deadline):
            stack.pop()
            if stack:
                _merge_into_parent(frame, stack[-1])
            continue

        if extending and frame.child_tokens is not None:
            extending = False                     # this level would start its second child
            if not clock() < end_time:
                continue

        batch_tokens, batch_scores, alive = [], [], 0
        for i in range(n):
            if frame.candidates[i]:
                score, t = frame.candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                alive += 1
            else:
                batch_tokens.append(pad_id)
                batch_scores.append(DEAD_SCORE)
        if alive == 0:
            extending = False
            stack.pop()
            if stack:
                _merge_into_parent(frame, stack[-1])
            continue

        level_cache = frame.cache
        if prepare_cache is not None:
            level_cache = prepare_cache(level_cache, frame.pos)
        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), frame.pos, device=device),
            past_key_values=level_cache,
            return_dict=True,
            use_cache=True,
        )
        frame.child_tokens = batch_tokens
        cands, sufs = expand(outputs.logits[:, -1], batch_scores, frame.remaining - 1)
        if sufs:
            extending = False
        stack.append(_Frame(frame.pos + 1, outputs.past_key_values, frame.remaining - 1, cands, sufs))

    if trace is not None and clock() >= end_time:
        trace["deadline_hit"] = True
    return root.suffixes


@torch.no_grad()
def inference_turbo_dfs_iterative(model, prefix_tokens, max_new_tokens, max_score, end_time, *,
                                  arc_tokens, eos_id, pad_id, time_cap: float = 540.0,
                                  device_log_softmax: bool = True, prepare_cache=None,
                                  trace=None, clock=time.time, extend_until=None):
    """Drop-in replacement for the parent ``inference_turbo_dfs``."""
    device = getattr(model, "device", None)
    input_ids = torch.tensor(prefix_tokens, device=device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs_iterative(
        model, outputs.logits[:, -1], max_new_tokens, max_score, [0.0] * input_ids.size(0),
        input_ids.size(1), outputs.past_key_values, clock(), end_time,
        arc_tokens=arc_tokens, eos_id=eos_id, pad_id=pad_id, time_cap=time_cap,
        device_log_softmax=device_log_softmax, prepare_cache=prepare_cache, trace=trace, clock=clock,
        extend_until=extend_until)
    return [(batch_id, sorted(beams, key=lambda x: x[0])) for batch_id, beams in suffixes.items()]
