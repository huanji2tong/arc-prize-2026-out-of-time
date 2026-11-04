"""Publication figures for the ARC Prize 2026 paper (all data read from frozen records)."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / 'paper/candidate_evidence_20261103'

# Muted, colour-blind-safe palette shared by every figure.
INK = '#1f2430'
SOFT = '#6b7280'
RULE = '#d7dbe2'
BLUE = '#2f6db5'      # length-aware / ours
TEAL = '#1b9e77'      # solved
GOLD = '#e6a23c'      # correct but unselected
SKY = '#9cc3e6'       # wrong candidates only
RED = '#c8553d'       # no candidate / unreachable
GREY = '#b8bec8'      # parent / control
ARC = ['#000000', '#1E93FF', '#F93C31', '#4FCC30', '#FFDC00', '#999999', '#E53AA3', '#FF851B', '#87D8F1', '#921231']
FULL = 6.9


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.4, 'axes.titleweight': 'bold',
        'xtick.labelsize': 6.4, 'ytick.labelsize': 6.4, 'legend.fontsize': 6.3,
        'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
        'axes.linewidth': .55, 'xtick.major.width': .55, 'ytick.major.width': .55,
        'xtick.major.size': 2.4, 'ytick.major.size': 2.4, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titlepad': 5, 'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.bbox': 'tight', 'savefig.pad_inches': .03, 'figure.dpi': 200})


def letter(ax, s, x=-0.02, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')


def save(fig, folder, name):
    fig.savefig(folder / f'{name}.pdf')
    fig.savefig(folder / f'{name}.png', dpi=300)
    plt.close(fig)


def draw_grid(ax, grid, x0, y0, size):
    g = np.array(grid)
    h, w = g.shape
    cell = size / max(h, w)
    ax.imshow(g, cmap=ListedColormap(ARC), vmin=0, vmax=9, interpolation='nearest',
              extent=(x0, x0 + w * cell, y0, y0 + h * cell), zorder=2)
    ax.add_patch(Rectangle((x0, y0), w * cell, h * cell, fill=False, lw=.6, ec=INK, zorder=3))
    return w * cell, h * cell


def teaser(folder, lb):
    """Figure 1: a long answer, two decoding timelines from real telemetry, and where answers go."""
    task = 'aa4ec2a5'
    ch = json.loads((ROOT / 'data/kaggle_2026/arc-agi_evaluation_challenges.json').read_text())
    so = json.loads((ROOT / 'data/kaggle_2026/arc-agi_evaluation_solutions.json').read_text())
    tel = json.loads((EVID / f'length_aware_development/gpu_run/arc_artifacts/task_telemetry/{task}.json').read_text())
    grid_in, grid_out = ch[task]['test'][0]['input'], so[task][0]
    L = len(grid_out) * len(grid_out[0]) + len(grid_out)
    fig = plt.figure(figsize=(FULL, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 2.25, 1.05], wspace=.32, left=.01, right=.99, top=.88, bottom=.17)

    ax = fig.add_subplot(gs[0]); ax.axis('off'); ax.set_xlim(0, 2.25); ax.set_ylim(-.35, 1.25); ax.set_aspect('equal')
    draw_grid(ax, grid_in, 0.0, 0.1, 1.0)
    draw_grid(ax, grid_out, 1.25, 0.1, 1.0)
    ax.add_patch(FancyArrowPatch((1.02, .58), (1.23, .58), arrowstyle='-|>', mutation_scale=7, lw=.7, color=INK))
    ax.text(.5, 1.13, 'test input', ha='center', fontsize=6.6, color=SOFT)
    ax.text(1.75, 1.13, 'answer', ha='center', fontsize=6.6, color=SOFT)
    h, w = len(grid_out), len(grid_out[0])
    ax.text(1.125, -.12, f'{h}$\\times${w} cells  $\\rightarrow$  $L={L}$ tokens', ha='center', fontsize=6.8)
    ax.text(1.125, -.30, f'task {task}', ha='center', fontsize=6, color=SOFT, family='monospace')
    letter(ax, 'a', x=0.04, y=0.98)

    ax = fig.add_subplot(gs[1])
    budget = tel['arms']['control']['requested_seconds']
    rows = [('length_aware', 'Length-aware schedule: the first views finish their grids', BLUE),
            ('control', 'Equal slices: every view is stopped mid-grid', GREY)]
    for yi, (arm, label, col) in enumerate(rows):
        t = 0.0
        for b in tel['arms'][arm]['batches']:
            dur = b['seconds']
            if b.get('skipped_exhausted_output_budget') or dur < 0.5:
                continue
            ok = b.get('raw_finished', 0) > 0
            ax.add_patch(Rectangle((t, yi - .27), dur, .54, fc=col if ok else 'white', ec=col, lw=.9,
                                   hatch=None if ok else '////', zorder=2))
            calls = b['dfs_trace'].get('prefix_calls', 0)
            if dur > 20:
                txt = (f'{calls} steps\n{b["raw_finished"]} grids' if ok else f'{calls} of {L} steps')
                ax.text(t + dur / 2, yi, txt, ha='center', va='center', fontsize=5.8, color='white' if ok else INK,
                        zorder=4, linespacing=1.05, bbox=None if ok else dict(fc='white', ec='none', pad=.8))
            ax.text(t + dur - 1.2, yi + .27 - .02, '\u2713' if ok else '\u2717', ha='right', va='top', fontsize=7,
                    color='white' if ok else RED, fontweight='bold', family='DejaVu Sans', zorder=5)
            t += dur
        ax.text(0, yi + .36, label, ha='left', va='bottom', fontsize=6.4, color=col if col != GREY else INK)
    tg = L * lb['timing']['median_seconds_per_call']
    ax.plot([tg, tg], [-.45, 1.62], color=RED, lw=.7, ls=(0, (2, 2)), zorder=1)
    ax.text(tg + 1.5, -.42, f'one greedy path: $L\\cdot t\\approx${tg:.0f} s', fontsize=5.8, color=RED, va='bottom')
    for k in range(1, 4):
        ax.plot([budget * k / 4] * 2, [.66, .74], color=SOFT, lw=.6)
    ax.set_xlim(0, budget); ax.set_ylim(-.5, 1.72)
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xticks([0, 45, 90, 135, 180]); ax.set_xlabel('decoding time for this test output (s)')
    ax.set_title('Same adapter, same 180 s, two schedules', loc='left')
    letter(ax, 'b', x=-0.03, y=1.03)

    ax = fig.add_subplot(gs[2])
    c = lb['confirmation']; rws = c['rows']
    reach = [r for r in rws if r['reachable_equal']]
    parts = [('Solved', sum(r['solved'] for r in rws), TEAL),
             ('Correct grid not chosen', sum(r['oracle'] - r['solved'] for r in rws), GOLD),
             ('Only wrong grids', sum(r['records'] > 0 and not r['oracle'] for r in reach), SKY),
             ('Fits slice, no grid', sum(r['records'] == 0 for r in reach), GREY),
             ('Exceeds slice', sum(not r['reachable_equal'] for r in rws), RED)]
    bottom = 0
    for label, n, col in parts:
        ax.bar([0], [n], bottom=bottom, color=col, width=.55, ec='white', lw=.6)
        ax.text(.36, bottom + n / 2, f'{n}  {label}', va='center', fontsize=6.0)
        bottom += n
    ax.set_xlim(-.6, 2.6); ax.set_ylim(0, len(rws)); ax.set_xticks([])
    ax.set_yticks([0, 20, 40, 60, len(rws)]); ax.spines['bottom'].set_visible(False)
    ax.set_ylabel(f'confirmation outputs (n={len(rws)})')
    ax.set_title('Where the answers go', loc='left')
    letter(ax, 'c', x=-0.08, y=1.03)
    save(fig, folder, 'teaser')


def clock(folder, lb):
    s = lb['timing']['median_seconds_per_call']
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.2), gridspec_kw={'width_ratios': [1.15, 1, 1.1], 'wspace': .38})
    lim = 1000
    def frame(ax):
        ax.fill_between([0, lim], [0, lim], 0, color=RED, alpha=.07, lw=0, zorder=0)
        ax.plot([0, lim], [0, lim], color=INK, lw=.6, ls=(0, (3, 2)), zorder=1)
        ax.set(xlim=(0, lim), ylim=(0, lim), xticks=[0, 250, 500, 750, 1000], yticks=[0, 250, 500, 750, 1000])
        ax.grid(color=RULE, lw=.4, zorder=0); ax.set_axisbelow(True)
        ax.set_xlabel('answer length $L$ (tokens)')
    ax = axs[0]; frame(ax)
    st = {'solved': (TEAL, 'o', 'solved'), 'missed': (GOLD, 'o', 'correct grid not chosen'),
          'wrong': (SKY, 'o', 'only wrong grids'), 'empty': (RED, 'X', 'no candidate')}
    seen = {}
    for r in lb['confirmation']['rows']:
        k = 'solved' if r['solved'] else 'missed' if r['oracle'] else 'empty' if r['records'] == 0 else 'wrong'
        key = round(r['greedy_reach_tokens']); j = seen.get(key, 0); seen[key] = j + 1
        y = r['greedy_reach_tokens'] + ((j % 5) - 2) * 14
        col, mk, _ = st[k]
        ax.scatter(r['tokens'], y, s=13, marker=mk, color=col, edgecolor='white' if mk == 'o' else 'none', lw=.4, zorder=3)
    un, ok = lb['confirmation']['unreachable_equal'], lb['confirmation']['reachable_equal']
    ax.text(985, 70, f"{un['outputs']} outputs below the line\n{un['oracle']} correct candidates", ha='right', fontsize=5.9, color=RED)
    ax.text(25, 720, f"{ok['outputs']} outputs above the line:\nall {ok['oracle']} correct candidates", ha='left', va='center', fontsize=5.9, color=INK)
    ax.set_ylabel('greedy reach of one slice, $\\tau/t$ (tokens)')
    ax.set_title('Confirmation panel (360 s per task)', loc='left'); letter(ax, 'a')
    handles = [plt.Line2D([], [], marker=m, ls='', color=c, markeredgecolor='white' if m == 'o' else c, markersize=4.2, label=l)
               for c, m, l in st.values()]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.22), ncol=2, handletextpad=.2, columnspacing=.8)
    ax = axs[1]; frame(ax)
    for r in lb['training_panel']['outputs']:
        on = r['nonempty_in_arms'] > 0
        ax.scatter(r['tokens'], r['greedy_reach_tokens'], s=20, marker='o' if on else 'X',
                   color=BLUE if on else RED, edgecolor='white' if on else 'none', lw=.4, zorder=3)
    ax.set_title('Weight-training panel (180 s)', loc='left'); letter(ax, 'b')
    handles = [plt.Line2D([], [], marker='o', ls='', color=BLUE, markeredgecolor='white', markersize=4.5, label='candidates in all 5 arms'),
               plt.Line2D([], [], marker='X', ls='', color=RED, markersize=4.5, label='empty in all 5 arms')]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.22), ncol=1, handletextpad=.2)
    ax = axs[2]
    ch = json.loads((ROOT / 'data/kaggle_2026/arc-agi_evaluation_challenges.json').read_text())
    so = json.loads((ROOT / 'data/kaggle_2026/arc-agi_evaluation_solutions.json').read_text())
    Ls = np.sort([len(g) * len(g[0]) + len(g) for t in ch for g in so[t]])
    surv = 1 - np.arange(len(Ls)) / len(Ls)
    ax.fill_between(Ls, surv, step='post', color=BLUE, alpha=.10, lw=0)
    ax.step(Ls, surv, where='post', color=BLUE, lw=1.1)
    for m, col, dy in [(1, TEAL, 0), (2, GOLD, 0), (3, RED, 0)]:
        r = 360 / (4 * m) / s; frac = float(np.mean(Ls > r))
        ax.axvline(r, color=col, lw=.8, ls=(0, (2, 1.5)))
        ax.scatter([r], [frac], s=14, color=col, zorder=4, edgecolor='white', lw=.4)
        ax.text(r + 14, frac + .02, f'{100 * frac:.0f}%', fontsize=6.4, color=col, fontweight='bold', va='bottom')
        ax.text(r + 14, frac + .02, f'{m} output{"s" if m > 1 else ""}/task', fontsize=5.4, color=col, va='top')
    ax.set(xlim=(0, 960), ylim=(0, 1.02), xlabel='answer length $L$ (tokens)', ylabel='share of outputs longer than $L$')
    ax.grid(color=RULE, lw=.4); ax.set_axisbelow(True)
    ax.set_title(f'All ARC-AGI-2 eval outputs (n={len(Ls)})', loc='left'); letter(ax, 'c')
    save(fig, folder, 'clock')


def pilot_fig(folder, pilot):
    if pilot is None:
        return False
    arms = [('control', 'Parent'), ('iterative', 'Iterative'), ('length_aware', 'Length-aware')]
    solved = {a: {f"{r['task_id']}_{r['test_index']}": r['pass2'] for r in pilot['metrics'][a + '_baseline']['per_output']}
              for a, _ in arms}
    so = json.loads((ROOT / 'data/kaggle_2026/arc-agi_evaluation_solutions.json').read_text())
    def tokens(key):
        t, i = key.rsplit('_', 1); g = so[t][int(i)]
        return len(g) * len(g[0]) + len(g)
    keys = sorted(pilot['per_output'], key=tokens)
    fig = plt.figure(figsize=(FULL, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.15, .8], wspace=.55, left=.11, right=.99, top=.86, bottom=.2)
    ax = fig.add_subplot(gs[0])
    for i, k in enumerate(keys):
        for j, (a, _) in enumerate(arms):
            d = pilot['per_output'][k][a]
            n = d['grids']
            col = '#ffffff' if n == 0 else plt.cm.Blues(0.25 + 0.6 * min(n, 8) / 8)
            ax.add_patch(Rectangle((j, i), .94, .86, fc=col, ec=RULE if n == 0 else 'none', lw=.5))
            if n == 0:
                ax.text(j + .47, i + .43, '\u2013', ha='center', va='center', fontsize=6.5, color=SOFT)
            else:
                mark = '\u2605' if solved[a][k] else ('\u25cf' if d['oracle'] else '')
                ax.text(j + .30, i + .43, str(n), ha='center', va='center', fontsize=6.4,
                        color='white' if n >= 5 else INK)
                if mark:
                    ax.text(j + .70, i + .43, mark, ha='center', va='center', fontsize=7.2 if mark == '\u2605' else 5.5,
                            color=GOLD if mark == '\u2605' else INK, family='DejaVu Sans')
    ax.set_xlim(0, 3); ax.set_ylim(len(keys), 0)
    ax.set_xticks([.47, 1.47, 2.47]); ax.set_xticklabels([l for _, l in arms]); ax.xaxis.tick_top()
    ax.set_yticks([i + .43 for i in range(len(keys))])
    ax.set_yticklabels([f"{k.split('_')[0]}{'/' + k.split('_')[1] if sum(x.split('_')[0] == k.split('_')[0] for x in keys) > 1 else '  '}  L={tokens(k)}" for k in keys], fontsize=5.6, family='monospace')
    ax.tick_params(length=0); [ax.spines[s].set_visible(False) for s in ['left', 'bottom']]
    ax.set_title('Distinct grids per output', loc='left', pad=14)
    ax.text(0, len(keys) + .75, '\u2605 solved   \u25cf correct grid in pool   \u2013 no candidate', fontsize=5.8, color=SOFT, family='DejaVu Sans')
    letter(ax, 'a', x=-0.33)
    ax = fig.add_subplot(gs[1])
    n = pilot['test_outputs']
    metrics = [('nonempty', [n - pilot['empty_outputs'][a] for a, _ in arms]),
               ('coverage', [pilot['candidate_oracle'][a] for a, _ in arms]),
               ('pass@2', [pilot['correct'][a + '_baseline'] for a, _ in arms])]
    cols = [GREY, SKY, BLUE]
    x = np.arange(len(metrics))
    for j in range(3):
        vals = [m[1][j] for m in metrics]
        ax.bar(x + (j - 1) * .26, vals, width=.24, color=cols[j], label=arms[j][1], zorder=2)
        for xi, v in zip(x, vals):
            ax.text(xi + (j - 1) * .26, v + .12, str(v), ha='center', fontsize=5.8)
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics]); ax.set_ylim(0, n)
    ax.set_ylabel(f'outputs (of {n})'); ax.grid(axis='y', color=RULE, lw=.4); ax.set_axisbelow(True)
    ax.legend(loc='upper right', handlelength=1, handletextpad=.3, borderaxespad=.1)
    ax.set_title('Same budget, more answers', loc='left'); letter(ax, 'b')
    ax = fig.add_subplot(gs[2])
    ms = [1000 * pilot['timings'][a]['seconds_per_forward_call'] for a, _ in arms]
    ax.barh(range(3), ms, color=cols, height=.6, zorder=2)
    for i, v in enumerate(ms): ax.text(v + 1.5, i, f'{v:.1f}', va='center', fontsize=6)
    ax.set_yticks(range(3)); ax.set_yticklabels([l for _, l in arms]); ax.invert_yaxis()
    ax.set_xlim(0, max(ms) * 1.3); ax.set_xlabel('ms per forward pass'); ax.grid(axis='x', color=RULE, lw=.4); ax.set_axisbelow(True)
    ax.set_title('Step cost', loc='left'); letter(ax, 'c', x=-0.45)
    save(fig, folder, 'pilot')
    return True


def confirmation_fig(folder, conf):
    if conf is None:
        return False
    keys = sorted(conf['per_output'], key=lambda k: conf['per_output'][k]['tokens'])
    solved = {a: {f"{r['task_id']}_{r['test_index']}": r['pass2'] for r in conf['metrics'][a + '_baseline']['per_output']}
              for a in ['control', 'length_aware']}
    fig, ax = plt.subplots(figsize=(3.35, 1.9))
    for k in keys:
        L = conf['per_output'][k]['tokens']
        c0, c1 = conf['per_output'][k]['control'], conf['per_output'][k]['length_aware']
        s0, s1 = solved['control'].get(k, 0), solved['length_aware'].get(k, 0)
        y0 = 0 if c0['grids'] == 0 else 1 if not c0['oracle'] else 2 + s0
        y1 = 0 if c1['grids'] == 0 else 1 if not c1['oracle'] else 2 + s1
        col = TEAL if y1 > y0 else RED if y1 < y0 else GREY
        ax.annotate('', xy=(L, y1 + .12), xytext=(L, y0 - .12 if y1 > y0 else y0 + .12),
                    arrowprops=dict(arrowstyle='-|>', lw=.8, color=col, mutation_scale=6)) if y1 != y0 else \
            ax.scatter([L], [y0], s=8, color=GREY, zorder=2)
    ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(['no grid', 'wrong grids', 'correct in pool', 'solved'])
    ax.set_xlabel('answer length $L$ (tokens)'); ax.set_xlim(0, 960); ax.set_ylim(-.5, 3.5)
    ax.grid(axis='y', color=RULE, lw=.4); ax.set_axisbelow(True)
    ax.set_title('Reserve tasks: parent $\\rightarrow$ length-aware', loc='left')
    save(fig, folder, 'confirmation')
    return True


def gallery(folder, pilot):
    """Appendix figure: demonstrations, test input and the submitted grid for the newly solved outputs."""
    if pilot is None:
        return False
    sub = json.loads((EVID / 'length_aware_development/gpu_run/arc_artifacts/length_aware/submission_baseline.json').read_text())
    ch = json.loads((ROOT / 'data/kaggle_2026/arc-agi_evaluation_challenges.json').read_text())
    so = json.loads((ROOT / 'data/kaggle_2026/arc-agi_evaluation_solutions.json').read_text())
    keys = list(pilot['gains'])
    fig, axs = plt.subplots(len(keys), 5, figsize=(FULL, 1.5 * len(keys)), gridspec_kw={'wspace': .08, 'hspace': .45})
    axs = np.atleast_2d(axs)
    for r, key in enumerate(keys):
        task, idx = key.rsplit('_', 1); idx = int(idx)
        demo = ch[task]['train'][0]
        guess = sub[task][idx]['attempt_1'] if sub[task][idx]['attempt_1'] == so[task][idx] else sub[task][idx]['attempt_2']
        att = 1 if sub[task][idx]['attempt_1'] == so[task][idx] else 2
        L = len(so[task][idx]) * len(so[task][idx][0]) + len(so[task][idx])
        panels = [(demo['input'], 'demo input'), (demo['output'], 'demo output'), (ch[task]['test'][idx]['input'], 'test input'),
                  (None, 'equal slices'), (guess, f'length-aware, guess {att}')]
        for c, (g, title) in enumerate(panels):
            ax = axs[r, c]; ax.set_xticks([]); ax.set_yticks([])
            if g is None:
                ax.text(.5, .5, 'no candidate\n(views stopped\nmid-grid)', ha='center', va='center', fontsize=6.3, color=RED, transform=ax.transAxes)
                for s in ax.spines.values(): s.set_visible(True); s.set_color(RULE); s.set_linestyle((0, (2, 2)))
            else:
                ax.imshow(np.array(g), cmap=ListedColormap(ARC), vmin=0, vmax=9, interpolation='nearest')
                for s in ax.spines.values(): s.set_visible(True); s.set_color(INK); s.set_linewidth(.5)
            ax.set_title(title, fontsize=6.4, fontweight='normal', color=SOFT if c < 3 else INK)
            ax.set_aspect('equal', adjustable='box')
        axs[r, 0].set_ylabel(f'{task}\nL={L}', fontsize=6.4, family='monospace', rotation=0, ha='right', va='center', labelpad=8)
        axs[r, 4].text(1.04, .5, '\u2713 correct', transform=axs[r, 4].transAxes, fontsize=6.6, color=TEAL, va='center',
                       family='DejaVu Sans', fontweight='bold')
    save(fig, folder, 'gallery')
    return True
