"""Frozen-pool comparisons and duplicate perturbations, with task-level pairing."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('ARC_DATA_DIR', str(ROOT/'data')))
if not (DATA_DIR/'arc-agi_evaluation_challenges.json').exists():
    DATA_DIR = Path('/Users/tom/.cache/arc2026/data')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'src/neural_runtime')]
from arc_decoder import score_kgmon
from neural_evidence import (grid_tuple, family_select, anchored_family_select,
                             score_submissions)
from two_guess_selection import likelihood_rank, select_coverage, select_support_coverage


def selectors(guesses):
    baseline = [grid_tuple(y) for y in score_kgmon(guesses)]
    anchor = baseline[0] if baseline else None
    return {
        'baseline': baseline[:2],
        'likelihood_only': likelihood_rank(guesses)[:2],
        'token_likelihood': likelihood_rank(guesses, True)[:2],
        'family': family_select(guesses)[:2],
        'anchored_family': anchored_family_select(guesses, baseline),
        'anchored_support_cover': select_support_coverage(guesses, anchor) if anchor else [],
        'anchored_soft_cover': select_coverage(guesses, anchor),
        'soft_cover': select_coverage(guesses),
    }


def pair(values):
    yy = [grid_tuple(y) for y in values[:2]]
    if not yy:
        yy = [((0,),)]
    return tuple(yy+[((0,),)] if len(yy) == 1 else yy)


def load_trace(path):
    result = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            row = json.loads(line)
            result[row['task_output']] = {
                r['source_key']: {k: v for k, v in r.items() if k != 'source_key'}
                for r in row['candidates']}
    # A missing candidate row is still an evaluated output. Never silently
    # exclude unsolved/no-candidate tasks from the accuracy denominator.
    manifest_path = next((p/'manifest.json' for p in [path.parent, path.parent.parent]
                          if (p/'manifest.json').exists()), None)
    selected_path = next((p/'selected_task_ids.json' for p in [path.parent, path.parent.parent]
                          if (p/'selected_task_ids.json').exists()), None)
    if selected_path:
        ids = json.loads(selected_path.read_text())
    elif manifest_path:
        manifest = json.loads(manifest_path.read_text())
        ids = manifest.get('completed_task_ids') or manifest.get('task_ids')
        if not ids:
            raise ValueError(f'No task IDs for denominator: {path}')
    else:
        raise ValueError(f'No manifest for denominator: {path}')
    challenges = json.loads((DATA_DIR/'arc-agi_evaluation_challenges.json').read_text())
    for tid in ids:
        for i in range(len(challenges[tid]['test'])):
            result.setdefault(f'{tid}_{i}', {})
    return result


def bootstrap(rows, name, reps=5000):
    by_task = defaultdict(list)
    for row in rows:
        by_task[row['task_id']].append(row[name]['pass2']-row['baseline']['pass2'])
    ids = sorted(by_task)
    deltas = np.array([sum(by_task[t]) for t in ids])
    sizes = np.array([len(by_task[t]) for t in ids])
    rng = np.random.default_rng(20261103)
    ix = rng.integers(len(ids), size=(reps, len(ids)))
    values = deltas[ix].sum(1)/sizes[ix].sum(1)
    return {'delta': float(deltas.sum()/sizes.sum()),
            'ci95': np.quantile(values, [.025, .975]).tolist(),
            'tasks': len(ids), 'replicates': reps}


def analyze(trace, labels, perturb=True):
    trace = trace.resolve()
    pools = load_trace(trace)
    frozen = {key: selectors(g) for key, g in pools.items()}
    # Selections are complete before opening target labels.
    truth = json.loads(labels.read_text())
    rows = []
    names = list(next(iter(frozen.values())))
    started = time.perf_counter()
    duplicates = {m: {'changed_pairs': 0, 'changed_sets': 0} for m in names}
    perturbations = 0
    for key, guessed in pools.items():
        tid, ix = key.rsplit('_', 1)
        target = grid_tuple(truth[tid][int(ix)])
        unique = {grid_tuple(r['solution']) for r in guessed.values()}
        row = {'task_output': key, 'task_id': tid, 'records': len(guessed),
               'unique_grids': len(unique), 'oracle': int(target in unique),
               'has_token_counts': all('score_aug_tokens' in r for r in guessed.values())}
        for name, selected in frozen[key].items():
            pp = pair(selected)
            row[name] = {'pass1': int(pp[0] == target), 'pass2': int(target in pp)}
        rows.append(row)
        if perturb:
            for source, record in guessed.items():
                cloned = dict(guessed)
                # Prefix before the first dot is a task identifier, not a view.
                prefix, sep, rest = source.partition('.')
                for j in range(8):
                    cloned[f'{prefix}.copyclone{j}'+(sep+rest if sep else '')] = record
                trial = selectors(cloned)
                perturbations += 1
                for name in names:
                    before, after = pair(frozen[key][name]), pair(trial[name])
                    duplicates[name]['changed_pairs'] += before != after
                    duplicates[name]['changed_sets'] += set(before) != set(after)
    metrics = {}
    for name in names:
        gains = [r['task_output'] for r in rows if r[name]['pass2'] > r['baseline']['pass2']]
        losses = [r['task_output'] for r in rows if r[name]['pass2'] < r['baseline']['pass2']]
        metrics[name] = {'pass1_correct': sum(r[name]['pass1'] for r in rows),
                         'pass2_correct': sum(r[name]['pass2'] for r in rows),
                         'gains': gains, 'losses': losses,
                         'paired_task_bootstrap': bootstrap(rows, name)}
    return {'trace': str(trace.relative_to(ROOT)),
            'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
            'tasks': len({r['task_id'] for r in rows}), 'outputs': len(rows),
            'oracle_correct': sum(r['oracle'] for r in rows), 'metrics': metrics,
            'clone_perturbations': perturbations, 'duplicate_sensitivity': duplicates,
            'analysis_seconds': time.perf_counter()-started, 'per_output': rows}


def main():
    global DATA_DIR
    p = argparse.ArgumentParser()
    p.add_argument('--trace', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--no-perturb', action='store_true')
    p.add_argument('--data', type=Path, default=DATA_DIR)
    args = p.parse_args()
    DATA_DIR = args.data
    traces = ([args.trace] if args.trace else sorted((ROOT/'results').rglob('candidate_trace.jsonl.gz')))
    labels = DATA_DIR/'arc-agi_evaluation_solutions.json'
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, trace in enumerate(traces):
        result = analyze(trace, labels, not args.no_perturb)
        results.append(result)
        (args.out/f'pool_{i:02d}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({'trace': result['trace'], 'tasks': result['tasks'],
            'outputs': result['outputs'], 'oracle': result['oracle_correct'],
            'pass2': {k: v['pass2_correct'] for k, v in result['metrics'].items()}}, ensure_ascii=False), flush=True)
    (args.out/'all_pools.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
