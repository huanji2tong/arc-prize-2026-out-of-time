"""Relate missing candidates to answer length and the per-view decoding slice.

Inputs are frozen records only:
  * completion_development telemetry: forward passes and seconds per view batch;
  * holdout_analysis.json and holdout telemetry: 48 tasks / 68 outputs;
  * generation_diagnostics/per_output.csv: the 9-output weight-training panel;
  * official evaluation solutions for target shapes (read after predictions froze).

Outputs paper/candidate_evidence_20261103/length_budget/analysis.json.
No accuracy is estimated for the planned allocation; only greedy reachability.
"""
from pathlib import Path
import csv
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from length_budget import (allocate_task, answer_tokens, estimate_answer_tokens,  # noqa: E402
                           greedy_reachable, predict_output_shape)

BASE = ROOT / "paper/candidate_evidence_20261103"
OUT = BASE / "length_budget"


def step_time():
    folder = BASE / "completion_development/gpu_run/arc_artifacts/task_telemetry"
    per_task = []
    for path in sorted(folder.glob("*.json")):
        d = json.loads(path.read_text())
        bs = d["arms"]["control"]["batches"]
        calls = sum(b["dfs_trace"]["prefix_calls"] for b in bs)
        finished = sum(b["raw_finished"] for b in bs)
        seconds = sum(b["seconds"] for b in bs)
        per_task.append({"task_id": d["task_id"], "forward_calls": calls, "seconds": round(seconds, 2),
                         "seconds_per_call": seconds / calls, "finished_sequences": finished})
    # Tasks without finished sequences spend no time on candidate rescoring.
    pure = [t["seconds_per_call"] for t in per_task if t["finished_sequences"] == 0]
    return {"per_task": per_task, "pure_decode_tasks": len(pure),
            "median_seconds_per_call": statistics.median(pure),
            "range_seconds_per_call": [min(pure), max(pure)], "batch_rows": 4}


def bins(rows, key, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        r = [x for x in rows if lo <= x[key] < hi]
        out.append({"lo": lo, "hi": hi, "outputs": len(r), "empty": sum(x["records"] == 0 for x in r),
                    "oracle": sum(x["oracle"] for x in r), "solved": sum(x["solved"] for x in r)})
    return out


def main():
    OUT.mkdir(exist_ok=True)
    timing = step_time()
    s = timing["median_seconds_per_call"]
    challenges = json.loads((ROOT / "data/kaggle_2026/arc-agi_evaluation_challenges.json").read_text())
    solutions = json.loads((ROOT / "data/kaggle_2026/arc-agi_evaluation_solutions.json").read_text())

    # Confirmation panel: 360 s per task split equally over 4 view batches per output.
    h = json.loads((BASE / "holdout_analysis.json").read_text())
    tel = {}
    for path in (BASE / "holdout_run/arc_artifacts/task_telemetry").glob("*.json"):
        d = json.loads(path.read_text())
        tel[d["task_id"]] = d["arms"]["control"]
    task_outputs = {}
    for o in h["per_output"]:
        task_outputs.setdefault(o["task_id"], []).append(o)
    rows = []
    for task, outs in task_outputs.items():
        budget = tel[task]["requested_seconds"]
        total_batches = tel[task]["total_batches"]
        assert total_batches == 4 * len(outs), task
        equal_slice = budget / total_batches
        estimates = []
        for o in outs:
            idx = int(o["task_output"].rsplit("_", 1)[1])
            test_input = challenges[task]["test"][idx]["input"]
            est, rule = estimate_answer_tokens(challenges[task]["train"], test_input)
            estimates.append({"est_tokens": est, "rule": rule})
        plan = allocate_task(estimates, budget, s)
        for o, e, p in zip(outs, estimates, plan):
            idx = int(o["task_output"].rsplit("_", 1)[1])
            target = solutions[task][idx]
            tokens = answer_tokens(len(target), len(target[0]))
            rows.append({"output": o["task_output"], "tokens": tokens, "est_tokens": e["est_tokens"],
                         "shape_rule": e["rule"], "equal_slice_seconds": equal_slice,
                         "greedy_reach_tokens": equal_slice / s, "load": tokens * s / equal_slice,
                         "records": o["records"], "oracle": o["oracle"], "solved": o["baseline"]["pass2"],
                         "planned_batches": p["batches"], "planned_slice_seconds": p["slice_seconds"],
                         "reachable_equal": greedy_reachable(tokens, equal_slice, s),
                         "reachable_planned": greedy_reachable(tokens, p["slice_seconds"], s)})
    rows.sort(key=lambda r: r["load"])

    def table(sel):
        return {"outputs": len(sel), "empty": sum(r["records"] == 0 for r in sel),
                "oracle": sum(r["oracle"] for r in sel), "solved": sum(r["solved"] for r in sel)}

    reachable = [r for r in rows if r["reachable_equal"]]
    unreachable = [r for r in rows if not r["reachable_equal"]]
    rescued = [r for r in unreachable if r["reachable_planned"]]
    # First-descent extension: a batch can only extend while its greedy descent is alive and
    # it has no candidate, so it can trigger only where the slice is shorter than greedy
    # completion plus prefill.  It completes if greedy decoding fits in the output budget.
    prefill = 2.0
    eligible = [r for r in rows if r["tokens"] * s + prefill > r["equal_slice_seconds"]]
    extension = {"prefill_seconds_assumed": prefill, "eligible": len(eligible),
                 "eligible_completes_within_output_budget":
                     sum(r["tokens"] * s + prefill <= 4 * r["equal_slice_seconds"] for r in eligible),
                 "eligible_nonempty": sum(r["records"] > 0 for r in eligible),
                 "eligible_oracle": sum(r["oracle"] for r in eligible),
                 "eligible_solved": sum(r["solved"] for r in eligible),
                 "eligible_outputs": [r["output"] for r in eligible]}
    confirm = {"tasks": len(task_outputs), "outputs": len(rows), "budget_seconds_per_task": 360,
               "load_bins": bins(rows, "load", [0, .5, .8, 1.0, 10]),
               "token_bins": bins(rows, "tokens", [0, 200, 400, 600, 1000]),
               "reachable_equal": table(reachable), "unreachable_equal": table(unreachable),
               "empty_total": sum(r["records"] == 0 for r in rows),
               "planned_reachable_among_unreachable": len(rescued),
               "planned_batches_for_rescued": sorted({r["planned_batches"] for r in rescued}),
               "planned_reduces_views_on_reachable": sum(r["planned_batches"] < 4 for r in reachable),
               "planned_unreachable": sum(not r["reachable_planned"] for r in rows),
               "first_descent_extension": extension,
               "rows": rows}

    # Weight-training panel: 180 s per arm, equal slices; same nine outputs in all arms.
    seen = {}
    for r in csv.DictReader((BASE / "generation_diagnostics/per_output.csv").open()):
        seen.setdefault((r["task_id"], int(r["test_index"])), []).append(int(r["records"]))
    comp_tel = {json.loads(p.read_text())["task_id"]: json.loads(p.read_text())["arms"]["control"]
                for p in (BASE / "completion_development/gpu_run/arc_artifacts/task_telemetry").glob("*.json")}
    training = []
    for (task, idx), recs in sorted(seen.items()):
        target = solutions[task][idx]
        tokens = answer_tokens(len(target), len(target[0]))
        slice_s = comp_tel[task]["requested_seconds"] / comp_tel[task]["total_batches"]
        training.append({"output": f"{task}_{idx}", "tokens": tokens, "slice_seconds": slice_s,
                         "greedy_reach_tokens": slice_s / s, "reachable": greedy_reachable(tokens, slice_s, s),
                         "nonempty_in_arms": sum(x > 0 for x in recs), "arms": len(recs)})
    agree = sum((t["reachable"] and t["nonempty_in_arms"] > 0) or (not t["reachable"] and t["nonempty_in_arms"] == 0)
                for t in training)

    # Shape predictor on all public evaluation outputs (no model involved).
    rules, exact, over, under = {}, 0, 0, 0
    for task, item in challenges.items():
        for idx, test in enumerate(item["test"]):
            target = solutions[task][idx]
            shape, rule = predict_output_shape(item["train"], test["input"])
            est, _ = estimate_answer_tokens(item["train"], test["input"])
            true_tokens = answer_tokens(len(target), len(target[0]))
            rules.setdefault(rule, [0, 0])
            rules[rule][0] += 1
            if shape == (len(target), len(target[0])):
                rules[rule][1] += 1
                exact += 1
            over += est > true_tokens
            under += est < true_tokens
    n_eval = sum(v[0] for v in rules.values())
    predictor = {"outputs": n_eval, "exact_shape": exact, "over_estimate": over, "under_estimate": under,
                 "rules": {k: {"outputs": v[0], "exact": v[1]} for k, v in rules.items()}}

    result = {"timing": timing, "confirmation": confirm,
              "training_panel": {"outputs": training, "classification_agreement": agree,
                                 "n": len(training)},
              "shape_predictor": predictor,
              "notes": ["Greedy reachability is a necessary condition for any candidate; it does not imply correctness.",
                        "Step time is measured on the completion pilot control arm (4 rows per batch, L4).",
                        "Target shapes are read from official solutions after predictions were frozen."]}
    (OUT / "analysis.json").write_text(json.dumps(result, indent=1))
    c = confirm
    print(json.dumps({"seconds_per_call": round(s, 4), "load_bins": c["load_bins"],
                      "reachable": c["reachable_equal"], "unreachable": c["unreachable_equal"],
                      "rescued": c["planned_reachable_among_unreachable"],
                      "planned_unreachable": c["planned_unreachable"],
                      "views_reduced_on_reachable": c["planned_reduces_views_on_reachable"],
                      "extension": c["first_descent_extension"],
                      "training_agreement": f"{agree}/{len(training)}", "predictor": predictor}, indent=1))


if __name__ == "__main__":
    main()
