"""Assemble the open-source release and the Paper Track submission package.

Creates release/arc-prize-2026-out-of-time/ (GitHub-ready, no credentials, no model weights,
no competition data) and a zip next to it. Safe to re-run after new results arrive.
"""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EVID = 'paper/candidate_evidence_20261103'
DEST = ROOT / 'release/arc-prize-2026-out-of-time'

CODE = ['src/fast_dfs.py', 'src/length_budget.py', 'src/neural_evidence.py', 'src/two_guess_selection.py',
        'scripts/analyze_length_budget.py', 'scripts/build_prize_paper.py', 'scripts/prize_figures.py',
        'scripts/test_fast_dfs.py', 'scripts/test_length_budget.py', 'scripts/build_length_aware_pilot.py',
        'scripts/build_length_aware_confirmation.py', 'scripts/build_l1_submission.py', 'scripts/build_conditional_ttt.py',
        'scripts/analyze_two_guess.py', 'scripts/build_release_package.py', 'scripts/build_writeup.py']
UPSTREAM = ['notebooks/baseline_33_89_source.py', 'notebooks/baseline_33_89.ipynb']
NOTEBOOKS = ['kaggle_api/paper_versions/B0', 'kaggle_api/paper_versions/S1', 'kaggle_api/paper_versions/S2',
             'kaggle_api/paper_versions/L1', 'kaggle_api/length_aware_pilot', 'kaggle_api/length_aware_confirmation']
RECORD_FILES = ['length_budget/analysis.json', 'holdout_analysis.json', 'duplicate_analysis/pool_00.json',
                'conditional_ttt_development/analysis.json', 'context_weighted_ttt_development/analysis.json',
                'completion_development/analysis.json', 'live_repair_confirmation/analysis.json',
                'generation_diagnostics/per_output.csv', 'task_usage_registry.json',
                'version_study/B0_submission.json', 'version_study/S1_submission.json', 'version_study/S2_submission.json',
                'version_study/protocol.json', 'l1_submission/L1_submission.json', 'l1_submission/protocol.json',
                'length_aware_development/analysis.json', 'length_aware_development/protocol.json',
                'length_aware_confirmation/analysis.json', 'length_aware_confirmation/protocol.json']
RECORD_DIRS = ['holdout_run/arc_artifacts', 'completion_development/gpu_run/arc_artifacts',
               'length_aware_development/gpu_run/arc_artifacts', 'length_aware_confirmation/gpu_run/arc_artifacts']
PAPER = 'paper/arc_prize_2026_submission'


def copy(rel, base=ROOT):
    src = base / rel
    if not src.exists():
        return False
    dst = DEST / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.lock', '*.pid'))
    else:
        shutil.copy2(src, dst)
    return True


README = r'''# Out of Time: Where Test-Time-Trained ARC-AGI-2 Solvers Lose Their Answers

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
'''

NOTICE = '''# Attribution

* Upstream notebook: **Failed in AIMO**, Koushik Rudra, Kaggle scriptVersionId 312063593, Apache License 2.0
  (https://www.kaggle.com/code/koushikrudra/failed-in-aimo?scriptVersionId=312063593). Its solver, loader and
  decoder modules are embedded, with explicit patches, in the notebooks under `kaggle_api/`.
* Model: `sorokin/qwen3_4b_grids15_sft139` (NVARC, I. Sorokin and J.-F. Puget), referenced as a Kaggle input,
  not redistributed. Dependency: `sorokin/pip-install-unsloth-flash-patch`.
* All other code in this repository was written by Jingtao Huang and is released under MIT-0.
'''


def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    missing = []
    for rel in CODE + UPSTREAM + NOTEBOOKS:
        if not copy(rel):
            missing.append(rel)
    for rel in RECORD_FILES + RECORD_DIRS:
        if not copy(f'{EVID}/{rel}'):
            missing.append(f'{EVID}/{rel}')
    for rel in ['main.tex', 'main.pdf', 'numbers.tex', 'numbers_extra.tex', 'flags.tex', 'figures',
                'table_leaderboard.tex', 'table_selectors.tex', 'table_generator.tex', 'table_pilot.tex',
                'pilot_results.tex', 'confirmation_results.tex', 'kaggle_writeup.md', 'kaggle_card.png', '提交清单.md']:
        copy(f'{PAPER}/{rel}')
    shutil.copy2(ROOT / 'LICENSES/MIT-0.txt', DEST / 'LICENSE')
    (DEST / 'LICENSES').mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'LICENSES/Apache-2.0.txt', DEST / 'LICENSES/Apache-2.0.txt')
    (DEST / 'README.md').write_text(README)
    (DEST / 'NOTICE.md').write_text(NOTICE)
    (DEST / 'data/kaggle_2026').mkdir(parents=True, exist_ok=True)
    (DEST / 'data/kaggle_2026/README.md').write_text('Place the ARC-AGI-2 evaluation challenges and solutions from the Kaggle competition here.\n')
    (DEST / '.gitignore').write_text('__pycache__/\n*.pyc\ndata/kaggle_2026/*.json\n*.aux\n*.log\n*.out\n')
    # Safety: no credentials or tokens in the package.
    for f in DEST.rglob('*'):
        if f.is_file() and f.name != 'build_release_package.py' and f.suffix in {'.py', '.json', '.md', '.ipynb', '.tex', '.txt'}:
            text = f.read_text(errors='ignore')
            for bad in ['KAGGLE_KEY', 'access_token', 'refresh_token', '"username": "huangjt66", "key"']:
                assert bad not in text, (f, bad)
    manifest = {str(f.relative_to(DEST)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted(DEST.rglob('*')) if f.is_file()}
    (DEST / 'MANIFEST.sha256.json').write_text(json.dumps(manifest, indent=1))
    zpath = DEST.parent / (DEST.name + '.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(DEST.rglob('*')):
            if f.is_file():
                z.write(f, Path(DEST.name) / f.relative_to(DEST))
    size = sum(f.stat().st_size for f in DEST.rglob('*') if f.is_file())
    print(json.dumps({'files': len(manifest), 'bytes': size, 'zip': str(zpath), 'missing': missing}, indent=1))


if __name__ == '__main__':
    main()
