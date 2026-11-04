"""Equivalence tests: iterative DFS versus the verbatim parent recursive DFS.

The parent functions are extracted from notebooks/baseline_33_89_source.py and
executed unchanged with a deterministic mock model.  The mock cache is
functional (each forward returns a new history), and the mock asserts that the
position id equals the cache length, which is the invariant the Unsloth
inference path relies on.
"""
from pathlib import Path
import hashlib
import math
import re
import sys
import types
from collections import defaultdict

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fast_dfs import inference_turbo_dfs_iterative  # noqa: E402

VOCAB = 24
ARC_TOKENS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15]
EOS_ID, PAD_ID = 15, 13
MAX_SCORE = -math.log(0.2)


class FakeClock:
    """Advances by ``step`` seconds on every reading."""

    def __init__(self, step=0.0):
        self.now, self.step = 1000.0, step

    def time(self):
        self.now += self.step
        return self.now


def load_parent(clock):
    source = (ROOT / "notebooks/baseline_33_89_source.py").read_text()
    start = source.index("def turbo_dfs(")
    end = source.index("@torch.no_grad()\ndef calc_scores")
    code = source[start:end]
    assert "def inference_turbo_dfs" in code
    ns = {"torch": torch, "defaultdict": defaultdict, "time": types.SimpleNamespace(time=clock.time),
          "ARC_TOKENS": ARC_TOKENS, "EOS_ID": EOS_ID, "PAD_ID": PAD_ID}
    exec(compile(code, "parent_turbo_dfs", "exec"), ns)
    return ns["inference_turbo_dfs"]


class Out:
    def __init__(self, logits, cache):
        self.logits, self.past_key_values = logits, cache


class MockModel:
    """Deterministic language model over a tiny vocabulary.

    Logits depend on the full token history.  ``answer_len`` controls where an
    end token becomes likely; ``sharpness`` controls branching.
    """

    device = torch.device("cpu")

    def __init__(self, seed, answer_len, sharpness=5.0, eos_bias=7.0, tie_prob=0.35):
        self.seed, self.answer_len, self.sharpness, self.eos_bias = seed, answer_len, sharpness, eos_bias
        self.tie_prob = tie_prob
        self.calls = 0

    def _logits(self, hist, prefix_len):
        digest = hashlib.blake2b(repr((self.seed, hist[-6:], len(hist))).encode(), digest_size=8).digest()
        g = torch.Generator().manual_seed(int.from_bytes(digest, "little"))
        z = torch.randn(VOCAB, generator=g)
        body = [t for t in ARC_TOKENS if t != EOS_ID]
        first = body[int(torch.randint(len(body), (1,), generator=g))]
        z[first] += 2.0 * self.sharpness
        if torch.rand(1, generator=g).item() < self.tie_prob:   # occasional near-tie creates branches
            second = body[int(torch.randint(len(body), (1,), generator=g))]
            z[second] += 2.0 * self.sharpness - 4.0 * torch.rand(1, generator=g).item()
        generated = len(hist) - prefix_len
        z[EOS_ID] += 4.0 * self.eos_bias if generated >= self.answer_len else -self.eos_bias
        return z.to(torch.bfloat16)

    def __call__(self, input_ids, position_ids=None, past_key_values=None, return_dict=True, use_cache=True):
        self.calls += 1
        rows = input_ids.tolist()
        if past_key_values is None:
            hists = [(tuple(r), len(r)) for r in rows]
        else:
            hists = []
            for i, (hist, prefix_len) in enumerate(past_key_values):
                assert position_ids[i, 0].item() == len(hist), "position must equal cache length"
                hists.append((hist + tuple(rows[i]), prefix_len))
        logits = torch.stack([self._logits(h, p) for h, p in hists]).unsqueeze(1)
        return Out(logits, tuple(hists))


def canon(result):
    return [(b, [(s, list(t)) for s, t in beams]) for b, beams in result]


def run_pair(seed, n=4, prefix=5, answer_len=6, max_new=12, clock_step=0.0, budget=1e9,
             time_cap=540, sharpness=5.0, device_log_softmax=False):
    prefix_tokens = [[3 + (i + j) % 10 for j in range(prefix)] for i in range(n)]
    c1, c2 = FakeClock(clock_step), FakeClock(clock_step)
    parent = load_parent(c1)
    m1, m2 = MockModel(seed, answer_len, sharpness), MockModel(seed, answer_len, sharpness)
    end1, end2 = c1.now + budget, c2.now + budget
    a = parent(m1, prefix_tokens, max_new, MAX_SCORE, end1)
    b = inference_turbo_dfs_iterative(m2, prefix_tokens, max_new, MAX_SCORE, end2, arc_tokens=ARC_TOKENS,
                                      eos_id=EOS_ID, pad_id=PAD_ID, time_cap=time_cap,
                                      device_log_softmax=device_log_softmax, clock=c2.time)
    return canon(a), canon(b), m1.calls, m2.calls


def test_exhaustive_equivalence():
    nonempty = 0
    for seed in range(60):
        for sharp in (3.0, 5.0):
            a, b, ca, cb = run_pair(seed, sharpness=sharp, answer_len=4 + seed % 5)
            assert a == b, (seed, sharp)
            assert ca == cb
            nonempty += sum(bool(beams) for _, beams in a)
    assert nonempty > 50
    print(f"exhaustive: 120 configurations identical; {nonempty} nonempty row results")


def test_deadline_equivalence():
    # Each clock reading advances 1 s; deadlines cut the search at different depths.
    for seed in range(20):
        for budget in (3, 17, 60, 250):
            a, b, ca, cb = run_pair(seed, clock_step=1.0, budget=budget, sharpness=3.0)
            assert a == b and ca == cb, (seed, budget)
    print("deadline: 80 truncated searches identical")


def test_device_log_softmax_cpu():
    for seed in range(20):
        a, b, _, _ = run_pair(seed, device_log_softmax=True)
        assert a == b
    print("device log-softmax path identical on CPU")


def test_deep_answers():
    """A 1,100-token answer: the iterative search needs no recursion."""
    limit = sys.getrecursionlimit()
    model = MockModel(7, answer_len=1100, sharpness=9.0, eos_bias=40.0, tie_prob=0.0)
    clock = FakeClock(0.0)
    out = inference_turbo_dfs_iterative(model, [[3, 4, 5]], 1200, 5.0, clock.now + 1e9, arc_tokens=ARC_TOKENS,
                                        eos_id=EOS_ID, pad_id=PAD_ID, device_log_softmax=False,
                                        clock=FakeClock(0.0).time)
    assert out and any(len(t) == 1101 for _, t in out[0][1])
    parent = load_parent(FakeClock(0.0))
    failed = False
    try:
        parent(MockModel(7, answer_len=1100, sharpness=9.0, eos_bias=40.0, tie_prob=0.0), [[3, 4, 5]], 1200, 5.0, 1e12)
    except RecursionError:
        failed = True
    print(f"deep answer: iterative finds a 1,101-token suffix; recursive parent "
          f"{'raises RecursionError' if failed else 'completes'} at recursion limit {limit}")


def test_first_descent_extension():
    """Extension rescues long answers and is inert when a candidate fits in the slice."""
    rescued = 0
    for seed in range(20):
        prefix = [[3 + (i + j) % 10 for j in range(5)] for i in range(4)]
        kw = dict(arc_tokens=ARC_TOKENS, eos_id=EOS_ID, pad_id=PAD_ID, device_log_softmax=False)
        # 1) Long answer, short slice: the parent finds nothing.
        c0 = FakeClock(1.0)
        parent = load_parent(c0)
        base = parent(MockModel(seed, 60, 6.0, tie_prob=0.1), prefix, 80, MAX_SCORE, c0.now + 40)
        assert all(not beams for _, beams in base)
        c1 = FakeClock(1.0)
        ext = inference_turbo_dfs_iterative(MockModel(seed, 60, 6.0, tie_prob=0.1), prefix, 80, MAX_SCORE,
                                            c1.now + 40, clock=c1.time, extend_until=c1.now + 10_000, **kw)
        # Reference: unlimited search; every suffix found with extension must exist there.
        c2 = FakeClock(0.0)
        full = inference_turbo_dfs_iterative(MockModel(seed, 60, 6.0, tie_prob=0.1), prefix, 80, MAX_SCORE,
                                             1e18, clock=c2.time, **kw)
        full_set = {(b, s, tuple(t)) for b, beams in full for s, t in beams}
        found = [(b, s, tuple(t)) for b, beams in ext for s, t in beams]
        assert all(x in full_set for x in found)
        if any(beams for _, beams in full):
            rescued += bool(found)
        # 2) Short answer, generous slice: identical to the parent.
        a, b, _, _ = run_pair(seed, answer_len=5, budget=10_000, clock_step=1.0)
        c3 = FakeClock(1.0)
        e = inference_turbo_dfs_iterative(MockModel(seed, 5, 5.0), [[3 + (i + j) % 10 for j in range(5)] for i in range(4)],
                                          12, MAX_SCORE, c3.now + 10_000, clock=c3.time,
                                          extend_until=c3.now + 20_000, **kw)
        assert canon(e) == a
    assert rescued >= 15, rescued
    print(f"extension: {rescued}/20 long answers rescued; short answers unchanged")


if __name__ == "__main__":
    test_exhaustive_equivalence()
    test_deadline_equivalence()
    test_device_log_softmax_cpu()
    test_deep_answers()
    test_first_descent_extension()
    print("all fast_dfs tests passed")
