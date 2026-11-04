# Out of Time: Where Test-Time-Trained ARC-AGI-2 Solvers Lose Their Answers

Code, notebooks and frozen experiment records for the ARC Prize 2026 Paper Track submission by
**Jingtao Huang** (Kaggle: `huangjt66`). The paper is in `paper/arc_prize_2026_submission/main.pdf`.

Linked Kaggle entry (public): https://www.kaggle.com/code/huangjt66/arc2-paper-l1-length-aware-decoding  
Matched control (public): https://www.kaggle.com/code/huangjt66/arc2-paper-b0-matched-control

## What is here

| Path | Contents |
|---|---|
| `src/fast_dfs.py` | Exact iterative reimplementation of the solver's threshold DFS, plus the first-descent extension (`extend_until`). |
| `src/length_budget.py` | Answer-length estimates, the reachability bound and budget planning helpers. |
| `kaggle_api/paper_versions/L1` | **Linked Kaggle entry**: B0 with length-aware decoding (offline, 4x L4). |
| `kaggle_api/paper_versions/{B0,S1,S2}` | Matched control and the two selector variants. |
| `kaggle_api/length_aware_pilot`, `kaggle_api/length_aware_confirmation` | Matched pilot and reserve-task confirmation notebooks. |
| `paper/candidate_evidence_20261103/` | Frozen records: candidate pools, per-task telemetry, analyses, protocols, leaderboard receipts. |
| `scripts/` | Analysis, figure and paper builders, notebook builders, tests. |
| `notebooks/baseline_33_89*` | Upstream notebook (Koushik Rudra, *Failed in AIMO* V1), Apache-2.0, kept for attribution and for the equivalence test. |

## Reproduce every table and figure (CPU only)

```bash
pip install numpy matplotlib torch pymupdf      # pdflatex (TeX Live) is also required
# Place the ARC-AGI-2 evaluation files from the Kaggle competition in data/kaggle_2026/:
#   arc-agi_evaluation_challenges.json, arc-agi_evaluation_solutions.json
python scripts/test_fast_dfs.py                 # iterative DFS == parent DFS (200 searches) + extension tests
python scripts/test_length_budget.py
python scripts/analyze_length_budget.py         # decoding-clock analysis from frozen telemetry
python scripts/build_prize_paper.py             # figures, tables and PDF from frozen records
```

## Run the entries on Kaggle

Each notebook folder contains `kernel-metadata.json`. Inputs: competition `arc-prize-2026-arc-agi-2`,
model `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`, notebook output
`sorokin/pip-install-unsloth-flash-patch`; accelerator GPU L4 x4; internet off. A normal run checks four
public tasks; a competition rerun processes every hidden task.

## Results in brief

* Frozen pools, 48 held-out tasks / 68 outputs: eight selectors all solve 14; only 18 pools contain the answer.
* Decoding clock: none of the 13 outputs that cannot finish one greedy path within a view slice has a correct
  candidate; under a 360 s budget, 28% of all ARC-AGI-2 public evaluation outputs are undecodable.
* Matched pilot (same adapter, 180 s): empty pools 7/9 -> 1/9, coverage 1 -> 3, pass@2 0 -> 2, no losses.
* Hidden test set (post-deadline): B0 25.28, S1 22.78, S2 23.61; L1 see the paper.

## Licenses

New code and methods by the author: **MIT-0** (`LICENSE`). Upstream notebook modules embedded in the
Kaggle notebooks: **Apache-2.0**, attribution in `NOTICE.md`. Model weights and competition data are referenced,
not redistributed. Paper text and figures: CC BY 4.0.

## Citation

```bibtex
@misc{huang2026outoftime,
  title  = {Out of Time: Where Test-Time-Trained ARC-AGI-2 Solvers Lose Their Answers},
  author = {Huang, Jingtao},
  year   = {2026},
  note   = {ARC Prize 2026 Paper Track}
}
```
