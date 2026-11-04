"""Input-only replay selectors for the pinned public Qwen/NVARC candidate pool.

MIT-0. Original code. The baseline scores are reimplemented for comparison;
the upstream notebook retains its Apache-2.0 license and attribution.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import math
import statistics


def stable_seed(task_id, modulus=1024**2):
    return int.from_bytes(hashlib.blake2b(str(task_id).encode(),digest_size=8).digest(),'little') % modulus


def grid_tuple(grid):
    if hasattr(grid,'tolist'):grid=grid.tolist()
    if not isinstance(grid,(list,tuple)) or not 1<=len(grid)<=30:
        raise ValueError('grid must have 1..30 rows')
    width=len(grid[0]) if isinstance(grid[0],(list,tuple)) else 0
    if not 1<=width<=30:raise ValueError('grid must have 1..30 columns')
    result=[]
    for row in grid:
        if not isinstance(row,(list,tuple)) or len(row)!=width:raise ValueError('ragged grid')
        if any(isinstance(v,bool) or not isinstance(v,int) or v<0 or v>9 for v in row):
            raise ValueError('grid colors must be integers 0..9')
        result.append(tuple(row))
    return tuple(result)


def geometric_family(source_key):
    """Canonical D4 action, independent of color permutation and beam index."""
    a=((0,1,2),(3,4,5),(6,7,8))
    for op in source_key.split('.')[1:]:
        if op=='rot90':a=tuple(zip(*a))[::-1]
        elif op=='transpose':a=tuple(zip(*a))
        elif op.startswith(('permute','copy','out','ex','run')):continue
        else:raise ValueError(f'unknown augmentation descriptor: {op}')
    return tuple(v for row in a for v in row)


def clean_records(guesses):
    result=[]
    for source_key,g in guesses.items():
        try:
            y=grid_tuple(g['solution'])
            aug=tuple(float(x) for x in g['score_aug'])
            beam=float(g['beam_score'])
            if not aug or not all(math.isfinite(x) for x in aug+(beam,)):continue
            tokens=g.get('score_aug_tokens')
            if tokens is not None:
                if len(tokens)!=len(aug) or any(float(x)<=0 for x in tokens):continue
                normalized=statistics.fmean(x/float(n) for x,n in zip(aug,tokens))
            else:
                # For old traces, leave summed NLL unchanged. The field flags it.
                normalized=statistics.fmean(aug)
            family=geometric_family(source_key)
            result.append({'source':source_key,'grid':y,'family':family,'beam':beam,
                           'aug_mean':statistics.fmean(aug),'normalized_nll':normalized,
                           'has_token_counts':tokens is not None})
        except (KeyError,TypeError,ValueError,OverflowError):continue
    return result


def baseline_rank(guesses):
    """Original score_kgmon: number of beams minus mean augmented summed NLL."""
    groups=defaultdict(list)
    for r in clean_records(guesses):groups[r['grid']].append(r)
    scores={y:len(rs)-statistics.fmean(r['aug_mean'] for r in rs) for y,rs in groups.items()}
    return sorted(groups,key=lambda y:(-scores[y],y))


def family_rank(guesses):
    """Each geometric view supports an output once, regardless of beam clones.

    This is a fixed hypothesis about useful evidence, not a fitted probability.
    View support first; per-token likelihood breaks ties. Neither signal uses
    the hidden target. Rotated views are correlated, not independent samples.
    """
    groups=defaultdict(lambda:defaultdict(list))
    for r in clean_records(guesses):groups[r['grid']][r['family']].append(r['normalized_nll'])
    diagnostics=[]
    for y,by_family in groups.items():
        per_view=[min(xs) for xs in by_family.values()]
        diagnostics.append({'grid':y,'family_support':len(per_view),
                            'mean_per_view_nll':statistics.fmean(per_view)})
    return sorted(diagnostics,key=lambda r:(-r['family_support'],r['mean_per_view_nll'],r['grid']))


def anchored_family_select(guesses, baseline_outputs=None):
    """Keep baseline attempt 1; replace attempt 2 only with >=2-view evidence.

    It can improve or reduce pass@2; no monotonic-accuracy guarantee is made.
    The complete family ranking is clone-invariant; the anchored final selector
    inherits any clone sensitivity of the baseline anchor.
    """
    baseline=(baseline_rank(guesses) if baseline_outputs is None
              else [grid_tuple(y) for y in baseline_outputs])
    if not baseline:return []
    anchor=baseline[0]
    eligible=[r for r in family_rank(guesses) if r['grid']!=anchor and r['family_support']>=2]
    second=eligible[0]['grid'] if eligible else baseline[min(1,len(baseline)-1)]
    return [anchor,second]


def family_select(guesses):
    return [r['grid'] for r in family_rank(guesses)]


def validate_submission(challenges,submission):
    if set(submission)!=set(challenges):raise ValueError('task IDs do not match challenges')
    outputs=0
    for tid,t in challenges.items():
        rows=submission[tid]
        if len(rows)!=len(t['test']):raise ValueError(f'{tid}: test count mismatch')
        for row in rows:
            if set(row)!={'attempt_1','attempt_2'}:raise ValueError(f'{tid}: expected exactly two attempts')
            grid_tuple(row['attempt_1']);grid_tuple(row['attempt_2']);outputs+=1
    return {'tasks':len(challenges),'test_outputs':outputs,'valid':True}


def score_submissions(solutions,submissions):
    """Post-hoc scoring; this function is never called by a selector."""
    result={}
    for name,submission in submissions.items():
        rows=[]
        for tid,ys in solutions.items():
            for i,y in enumerate(ys):
                truth=grid_tuple(y);p=submission[tid][i]
                rows.append({'task_id':tid,'test_index':i,
                             'pass1':int(grid_tuple(p['attempt_1'])==truth),
                             'pass2':int(truth in (grid_tuple(p['attempt_1']),grid_tuple(p['attempt_2'])))})
        result[name]={'test_outputs':len(rows),'pass1':sum(r['pass1'] for r in rows)/len(rows),
                      'pass2':sum(r['pass2'] for r in rows)/len(rows),'per_output':rows}
    return result
