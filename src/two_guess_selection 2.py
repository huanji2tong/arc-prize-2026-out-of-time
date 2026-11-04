"""Duplicate-invariant view coverage for a two-output candidate budget. MIT-0.

The utilities below are selection surrogates, not calibrated correctness
probabilities. They read candidate records, never target grids.
"""
from collections import defaultdict
from itertools import combinations
import math
import statistics

from neural_evidence import clean_records, family_rank, grid_tuple


def likelihood_rank(guesses, normalize=False):
    grouped = defaultdict(list)
    for row in clean_records(guesses):
        grouped[row['grid']].append(row['normalized_nll' if normalize else 'aug_mean'])
    return sorted(grouped, key=lambda y: (statistics.fmean(grouped[y]), y))


def view_utilities(guesses, temperature=1.0):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    by_view = defaultdict(lambda: defaultdict(list))
    for row in clean_records(guesses):
        by_view[row['family']][row['grid']].append(row['normalized_nll'])
    utilities = {}
    for view, grids in sorted(by_view.items()):
        losses = {y: min(vals) for y, vals in grids.items()}
        best = min(losses.values())
        weights = {y: math.exp(-(loss-best)/temperature) for y, loss in losses.items()}
        z = sum(weights.values())
        utilities[view] = {y: w/z for y, w in weights.items()}
    return utilities


def coverage(pair, utilities):
    if not utilities:
        return 0.0
    return statistics.fmean(max((values.get(y, 0.0) for y in pair), default=0.0)
                            for values in utilities.values())


def select_coverage(guesses, anchor=None, temperature=1.0):
    utilities = view_utilities(guesses, temperature)
    order = [r['grid'] for r in family_rank(guesses)]
    if not order:
        return []
    preference = {y: i for i, y in enumerate(order)}
    if anchor is not None:
        first = grid_tuple(anchor)
        seconds = [y for y in order if y != first]
        if not seconds:
            return [first, first]
        second = min(seconds, key=lambda y: (-coverage((first, y), utilities), preference[y]))
        return [first, second]
    if len(order) == 1:
        return order*2
    # Enumerate all pairs; candidate pools are small and require no heuristic.
    pairs = combinations(order, 2)
    pair = min(pairs, key=lambda ab: (-coverage(ab, utilities),
                                    preference[ab[0]]+preference[ab[1]],
                                    preference[ab[0]], preference[ab[1]]))
    return list(pair)


def select_support_coverage(guesses, anchor):
    rows = clean_records(guesses)
    views = defaultdict(set)
    for r in rows:
        views[r['grid']].add(r['family'])
    first = grid_tuple(anchor)
    order = [r['grid'] for r in family_rank(guesses) if r['grid'] != first]
    if not order:
        return [first, first]
    pos = {y: i for i, y in enumerate(order)}
    second = min(order, key=lambda y: (-len(views[first] | views[y]), pos[y]))
    return [first, second]
