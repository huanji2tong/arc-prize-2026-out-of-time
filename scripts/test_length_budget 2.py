"""Unit tests for src/length_budget.py."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from length_budget import (allocate_task, answer_tokens, estimate_answer_tokens,  # noqa: E402
                           greedy_reachable, plan_view_batches, predict_output_shape, select_batches)


def grid(h, w):
    return [[0] * w for _ in range(h)]


def test_tokens():
    assert answer_tokens(30, 30) == 930      # parent max_new_tokens is this plus one
    assert answer_tokens(1, 1) == 2


def test_shape_rules():
    same = [{"input": grid(3, 4), "output": grid(3, 4)}, {"input": grid(5, 5), "output": grid(5, 5)}]
    assert predict_output_shape(same, grid(7, 2)) == ((7, 2), "same_as_input")
    const = [{"input": grid(3, 4), "output": grid(2, 2)}, {"input": grid(5, 5), "output": grid(2, 2)}]
    assert predict_output_shape(const, grid(9, 9)) == ((2, 2), "constant_output")
    scale = [{"input": grid(2, 3), "output": grid(4, 9)}, {"input": grid(3, 1), "output": grid(6, 3)}]
    assert predict_output_shape(scale, grid(5, 2)) == ((10, 6), "constant_scale")
    offset = [{"input": grid(2, 3), "output": grid(4, 3)}, {"input": grid(5, 1), "output": grid(7, 1)}]
    assert predict_output_shape(offset, grid(6, 6)) == ((8, 6), "constant_offset")
    mixed = [{"input": grid(2, 3), "output": grid(4, 3)}, {"input": grid(5, 1), "output": grid(1, 9)}]
    assert predict_output_shape(mixed, grid(6, 6)) == (None, "unknown")
    tokens, rule = estimate_answer_tokens(mixed, grid(6, 6))
    assert rule == "unknown" and tokens == answer_tokens(6, 9)


def test_planner():
    s = 0.1035
    assert plan_view_batches(4, 90, answer_tokens(30, 30), s) == 1
    assert plan_view_batches(4, 360, answer_tokens(10, 10), s) == 4
    assert plan_view_batches(0, 360, 10, s) == 0
    plan = allocate_task([{"est_tokens": 930}, {"est_tokens": 120}], 360, s)
    assert plan[0]["batches"] == 1 and plan[1]["batches"] == 4
    assert all(greedy_reachable(t, p["slice_seconds"], s) for t, p in zip([930, 120], plan))
    assert select_batches(list("abcd"), 2) == ["a", "c"]
    assert select_batches(list("abcd"), 5) == list("abcd")


def test_frozen_analysis():
    path = ROOT / "paper/candidate_evidence_20261103/length_budget/analysis.json"
    d = json.loads(path.read_text())
    c = d["confirmation"]
    assert c["outputs"] == 68 and c["reachable_equal"]["outputs"] + c["unreachable_equal"]["outputs"] == 68
    assert c["unreachable_equal"]["oracle"] == 0
    assert d["training_panel"]["classification_agreement"] == d["training_panel"]["n"] == 9
    assert c["first_descent_extension"]["eligible_oracle"] == 0


if __name__ == "__main__":
    test_tokens()
    test_shape_rules()
    test_planner()
    test_frozen_analysis()
    print("OK")
