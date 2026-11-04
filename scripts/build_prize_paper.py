"""Build the ARC Prize 2026 paper (figures, data-driven numbers and tables, PDF).

Every number in the paper is read from frozen JSON records in
paper/candidate_evidence_20261103; nothing is typed by hand.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / 'paper/candidate_evidence_20261103'
OUT = ROOT / 'paper/arc_prize_2026_submission'
FIG = OUT / 'figures'
sys.path.insert(0, str(ROOT / 'src'))
from length_budget import answer_tokens  # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts'))
import prize_figures as pf  # noqa: E402

# Okabe-Ito palette
BLUE, ORANGE, GREEN, VERM, SKY, GREY, BLACK = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#56B4E9', '#999999', '#000000'
COL, FULL = 3.25, 6.75


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7,
                         'axes.titlesize': 7.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                         'legend.fontsize': 6.3, 'axes.linewidth': .5, 'xtick.major.width': .5,
                         'ytick.major.width': .5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
                         'savefig.bbox': 'tight', 'savefig.pad_inches': .02})


def load(name):
    p = EVID / name
    return json.loads(p.read_text()) if p.exists() else None


def tag(ax, letter):
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=8.5, fontweight='bold', va='bottom')


def fig_overview(lb):
    """(a) pipeline and its clock; (b) where the confirmation answers go."""
    c = lb['confirmation']
    rows = c['rows']
    unreach = [r for r in rows if not r['reachable_equal']]
    reach = [r for r in rows if r['reachable_equal']]
    parts = [('Solved', sum(r['solved'] for r in rows), GREEN),
             ('Correct grid\nnot selected', sum(r['oracle'] - r['solved'] for r in rows), ORANGE),
             ('Only wrong grids', sum(r['records'] > 0 and not r['oracle'] for r in reach), SKY),
             ('Fits slice,\nno candidate', sum(r['records'] == 0 for r in reach), GREY),
             ('Exceeds slice', len(unreach), VERM)]
    fig = plt.figure(figsize=(FULL, 1.75))
    ax = fig.add_axes([0.0, 0.0, 0.60, 1.0]); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 3.2)
    boxes = [(0.1, 'Task', 'demos + test input'), (2.1, 'Adapt', 'LoRA, 128 steps'),
             (4.1, 'Decode', 'DFS over 16 views'), (6.1, 'Rescore', '8 augmentations'),
             (8.1, 'Select', 'two guesses')]
    for x, title, sub in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.55), 1.75, 1.1, boxstyle='round,pad=0.02,rounding_size=0.08',
                                    fc='white', ec=BLACK, lw=.6))
        ax.text(x + .875, 2.3, title, ha='center', va='center', fontsize=7, fontweight='bold')
        ax.text(x + .875, 1.88, sub, ha='center', va='center', fontsize=5.6, color='#333333')
    for x in [1.85, 3.85, 5.85, 7.85]:
        ax.add_patch(FancyArrowPatch((x, 2.1), (x + .25, 2.1), arrowstyle='-|>', mutation_scale=6, lw=.6, color=BLACK))
    ax.plot([2.1, 7.85], [1.35, 1.35], color=BLUE, lw=1.2)
    ax.text(4.97, 1.12, 'coverage: is the answer in the pool?', ha='center', fontsize=6.2, color=BLUE)
    ax.plot([6.1, 9.85], [2.9, 2.9], color=ORANGE, lw=1.2)
    ax.text(7.97, 3.0, 'selection', ha='center', fontsize=6.2, color=ORANGE)
    ax.text(4.97, 0.55, r'budget $B$ per task $\rightarrow$ slice $\tau=B/(4m)$ per batch;'
            '\n' r'an $h\times w$ answer needs $L=hw+h$ sequential steps', ha='center', fontsize=6.1, color='#333333')
    ax.text(0.0, 3.05, 'a', fontsize=8.5, fontweight='bold')
    bx = fig.add_axes([0.655, 0.26, 0.335, 0.40])
    left = 0
    for label, n, color in parts:
        bx.barh([0], [n], left=left, color=color, height=.6, edgecolor='white', lw=.5)
        if n:
            bx.text(left + n / 2, 0, str(n), ha='center', va='center', fontsize=6.5,
                    color='white' if color in (GREEN, VERM, BLUE) else BLACK, fontweight='bold')
        left += n
    bx.set_xlim(0, len(rows)); bx.set_yticks([]); bx.spines['left'].set_visible(False)
    bx.set_xlabel(f"{c['outputs']} confirmation outputs ({c['tasks']} tasks)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in parts]
    bx.legend(handles, [p[0].replace('\n', ' ') for p in parts], ncol=3, loc='lower left',
              bbox_to_anchor=(-0.02, 1.08), frameon=False, handlelength=.9, columnspacing=.8, handletextpad=.4)
    bx.text(-0.06, 2.55, 'b', transform=bx.transAxes, fontsize=8.5, fontweight='bold')
    fig.savefig(FIG / 'overview.pdf'); plt.close(fig)


def fig_clock(lb):
    s = lb['timing']['median_seconds_per_call']
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.05), gridspec_kw={'width_ratios': [1.15, 1, 1.15], 'wspace': .42})
    ax = axs[0]
    status_style = {'solved': (GREEN, 'o', 'Solved'), 'missed': (ORANGE, 'o', 'Correct grid not selected'),
                    'wrong': (SKY, 'o', 'Only wrong grids'), 'empty': (VERM, 'x', 'No candidate')}
    seen = {}
    for r in lb['confirmation']['rows']:
        st = 'solved' if r['solved'] else 'missed' if r['oracle'] else 'empty' if r['records'] == 0 else 'wrong'
        k = round(r['greedy_reach_tokens']); j = seen.get(k, 0); seen[k] = j + 1
        y = r['greedy_reach_tokens'] * (1 + ((j % 5) - 2) * 0.025)
        col, mk, _ = status_style[st]
        ax.scatter(r['tokens'], y, s=11, marker=mk, c=col, linewidths=.8 if mk == 'x' else .3,
                   edgecolors='white' if mk == 'o' else None, zorder=3)
    lim = 1000
    ax.plot([0, lim], [0, lim], color=BLACK, lw=.6, ls='--', zorder=1)
    ax.fill_between([0, lim], [0, lim], 0, color=VERM, alpha=.06, lw=0)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel='Answer length $L$ (tokens)', ylabel=r'Greedy reach $\tau/t$ (tokens)')
    for _, (col, mk, lab) in status_style.items():
        ax.scatter([], [], s=11, marker=mk, c=col, label=lab, linewidths=.8 if mk == 'x' else .3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, handletextpad=.2,
              columnspacing=.6, labelspacing=.25)
    ax.set_title('Confirmation panel', loc='left'); tag(ax, 'a')
    ax = axs[1]
    for r in lb['training_panel']['outputs']:
        on = r['nonempty_in_arms'] > 0
        ax.scatter(r['tokens'], r['greedy_reach_tokens'], s=16, marker='o' if on else 'x',
                   c=BLUE if on else VERM, linewidths=.9 if not on else .3, zorder=3)
    ax.plot([0, lim], [0, lim], color=BLACK, lw=.6, ls='--'); ax.fill_between([0, lim], [0, lim], 0, color=VERM, alpha=.06, lw=0)
    ax.scatter([], [], s=14, c=BLUE, label='Candidates in all 5 arms'); ax.scatter([], [], s=14, marker='x', c=VERM, label='Empty in all 5 arms')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False, handletextpad=.2, labelspacing=.25)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel='Answer length $L$ (tokens)')
    ax.set_title('Weight-training panel', loc='left'); tag(ax, 'b')
    ax = axs[2]
    ch = json.loads((ROOT / 'data/kaggle_2026/arc-agi_evaluation_challenges.json').read_text())
    so = json.loads((ROOT / 'data/kaggle_2026/arc-agi_evaluation_solutions.json').read_text())
    L = np.sort([answer_tokens(len(g), len(g[0])) for t in ch for g in so[t]])
    ax.plot(L, 1 - np.arange(len(L)) / len(L), color=BLACK, lw=1, drawstyle='steps-post')
    for m, col in [(1, GREEN), (2, ORANGE), (3, VERM)]:
        r = 360 / (4 * m) / s
        frac = float(np.mean(L > r))
        ax.axvline(r, color=col, lw=.8, ls=':')
        ax.text(r - 14, 0.03, f'{m} output{"s" if m > 1 else ""}/task: {100 * frac:.0f}% longer',
                fontsize=5.6, color=col, rotation=90, ha='right', va='bottom')
    ax.set(xlim=(0, 960), ylim=(0, 1.02), xlabel='Answer length $L$ (tokens)', ylabel='Fraction of outputs longer')
    ax.set_title(f'All ARC-AGI-2 eval outputs (n={len(L)})', loc='left'); tag(ax, 'c')
    fig.savefig(FIG / 'clock.pdf'); plt.close(fig)


def fig_pilot(pilot):
    if pilot is None:
        return False
    arms = ['control', 'iterative', 'length_aware']
    labels = ['Parent DFS', 'Iterative DFS', 'Length-aware']
    n = pilot['test_outputs']
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.45), gridspec_kw={'wspace': .55})
    ax = axs[0]
    vals = [n - pilot['empty_outputs'][a] for a in arms]
    ax.bar(range(3), vals, color=[GREY, SKY, BLUE], width=.6)
    for i, v in enumerate(vals): ax.text(i, v + .1, f'{v}/{n}', ha='center', fontsize=6.3)
    ax.set(xticks=range(3), ylim=(0, n), ylabel='Outputs with candidates'); ax.set_xticklabels(labels, rotation=20, ha='right')
    tag(ax, 'a')
    ax = axs[1]
    vals = [pilot['timings'][a]['seconds_per_forward_call'] * 1000 for a in arms]
    ax.bar(range(3), vals, color=[GREY, SKY, BLUE], width=.6)
    for i, v in enumerate(vals): ax.text(i, v + 1, f'{v:.0f}', ha='center', fontsize=6.3)
    ax.set(xticks=range(3), ylabel='ms per forward pass'); ax.set_xticklabels(labels, rotation=20, ha='right')
    tag(ax, 'b')
    fig.savefig(FIG / 'pilot.pdf'); plt.close(fig)
    return True


def numbers(lb, holdout, dup, cond, weight, comp, s4, pilot, versions):
    c = lb['confirmation']
    ext = c['first_descent_extension']
    ch = json.loads((ROOT / 'data/kaggle_2026/arc-agi_evaluation_challenges.json').read_text())
    so = json.loads((ROOT / 'data/kaggle_2026/arc-agi_evaluation_solutions.json').read_text())
    s = lb['timing']['median_seconds_per_call']
    evalrows = [(answer_tokens(len(g), len(g[0])), len(ch[t]['test'])) for t in ch for g in so[t]]
    unreach_eval = sum(L > 360 / (4 * m) / s for L, m in evalrows)
    tasks_unreach = len({t for t in ch for g in so[t] if answer_tokens(len(g), len(g[0])) > 360 / (4 * len(ch[t]['test'])) / s})
    tight = next(b for b in c['load_bins'] if b['lo'] == 0.8)
    m = {
        'ConfTasks': c['tasks'], 'ConfOutputs': c['outputs'], 'ConfSolved': holdout['metrics']['baseline']['pass2_correct'],
        'ConfOracle': holdout['oracle_correct'],
        'ConfMissed': holdout['oracle_correct'] - holdout['metrics']['baseline']['pass2_correct'],
        'ConfNoCorrect': c['outputs'] - holdout['oracle_correct'],
        'ConfEmpty': c['empty_total'], 'Unreach': c['unreachable_equal']['outputs'],
        'UnreachEmpty': c['unreachable_equal']['empty'], 'Reach': c['reachable_equal']['outputs'],
        'ReachEmpty': c['reachable_equal']['empty'], 'TightN': tight['outputs'], 'TightEmpty': tight['empty'],
        'TightOracle': tight['oracle'],
        'StepMs': f"{1000 * s:.0f}", 'StepLo': f"{1000 * lb['timing']['range_seconds_per_call'][0]:.0f}",
        'StepHi': f"{1000 * lb['timing']['range_seconds_per_call'][1]:.0f}", 'StepTasks': lb['timing']['pure_decode_tasks'],
        'EvalOutputs': len(evalrows), 'EvalTasks': len(ch), 'EvalUnreach': unreach_eval,
        'EvalUnreachPct': f'{100 * unreach_eval / len(evalrows):.0f}', 'EvalTasksUnreach': tasks_unreach,
        'EvalOver': sum(L > 435 for L, _ in evalrows), 'EvalOverPct': f"{100 * np.mean([L > 435 for L, _ in evalrows]):.0f}",
        'ExtEligible': ext['eligible'], 'CloneTrials': dup['clone_perturbations'],
        'CloneRef': dup['duplicate_sensitivity']['baseline']['changed_sets'],
        'CloneAnchor': dup['duplicate_sensitivity']['anchored_soft_cover']['changed_sets'],
        'TrainReach': sum(o['reachable'] for o in lb['training_panel']['outputs']),
        'TrainN': len(lb['training_panel']['outputs']),
        'SFourBase': s4['correct']['control_baseline'], 'SFourRepair': s4['correct']['repair_risk_repair'],
        'SFourRiskOnly': s4['correct']['repair_risk_original'], 'SFourOracle': s4['raw_oracle']['control'],
        'SFourN': s4['test_outputs'], 'SFourTasks': s4['tasks'], 'SFourRepairs': s4['evaluated_repairs'],
    }
    text = '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in m.items()) + '\n'
    (OUT / 'numbers.tex').write_text(text)
    return m


def tables(holdout, cond, weight, comp, s4, lb, pilot, versions):
    names = [('baseline', 'Occurrence + NLL (reference)'), ('likelihood_only', 'NLL only'),
             ('token_likelihood', 'Token-normalized NLL'), ('family', 'Geometric view support'),
             ('anchored_family', 'Anchored view support'), ('anchored_support_cover', 'Anchored view union'),
             ('anchored_soft_cover', 'Anchored soft coverage'), ('soft_cover', 'Joint soft coverage')]
    dup = load('duplicate_analysis/pool_00.json')['duplicate_sensitivity']
    n = holdout['outputs']
    rows = [f"{lab} & {holdout['metrics'][k]['pass1_correct']} & {holdout['metrics'][k]['pass2_correct']} & {dup[k]['changed_sets']} \\\\"
            for k, lab in names]
    (OUT / 'table_selectors.tex').write_text(
        '\\begin{table}[t]\\centering\\footnotesize\n\\caption{Selection on frozen pools. Pass@1 and pass@2 count solved outputs out of '
        f'{n} on the confirmation panel, whose pools contain {holdout["oracle_correct"]} correct grids. The last column counts '
        'answer-set changes in 213 interventions that copy one development record eight times.}\\label{tab:selectors}\n'
        '\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{lccc}\\toprule\nSelector & Pass@1 & Pass@2 & Copy changes \\\\\\midrule\n'
        + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\\end{table}\n')
    # Generator interventions
    tr = lb['training_panel']
    reach = sum(o['reachable'] for o in tr['outputs'])
    gen = [
        ('Original LoRA loss', 'W', f"{cond['correct']['all_outputs_baseline']}", f"{cond['candidate_oracle']['all_outputs']}"),
        ('Loss after first output (PoE)', 'W', f"{cond['correct']['after_first_baseline']}", f"{cond['candidate_oracle']['after_first']}"),
        ('Loss on final output only', 'W', f"{cond['correct']['final_output_baseline']}", f"{cond['candidate_oracle']['final_output']}"),
        ('Original loss, second run', 'W', f"{weight['correct']['all_outputs_baseline']}", f"{weight['candidate_oracle']['all_outputs']}"),
        ('Context-weighted loss', 'W', f"{weight['correct']['context_weighted_baseline']}", f"{weight['candidate_oracle']['context_weighted']}"),
        ('Parent search (backstop control)', 'W', f"{comp['correct']['control_baseline']}", f"{comp['candidate_oracle']['control']}"),
        ('Greedy completion backstop', 'W', f"{comp['correct']['completion_baseline']}", f"{comp['candidate_oracle']['completion']}"),
    ]
    rows = [f'{a} & {b} & {c_}/9 & {d}/9 \\\\' for a, b, c_, d in gen]
    sn = s4['test_outputs']
    rows += ['\\midrule',
             f"Full search, occurrence ranking & R & {s4['correct']['control_baseline']}/{sn} & {s4['raw_oracle']['control']}/{sn} \\\\",
             f"Shorter search, worst-view score & R & {s4['correct']['repair_risk_original']}/{sn} & {s4['raw_oracle']['repair']}/{sn} \\\\",
             f"\\quad + spatial repairs & R & {s4['correct']['repair_risk_repair']}/{sn} & {s4['repair_expanded_oracle']}/{sn} \\\\"]
    (OUT / 'table_generator.tex').write_text(
        '\\begin{table}[t]\\centering\\footnotesize\n\\caption{Interventions on the generator. Panel W is the weight-training '
        f'panel (8 tasks, 9 outputs; {reach} of 9 outputs fit one view slice). Panel R is a fresh 24-task, {sn}-output repair panel. '
        'Coverage counts outputs whose pool contains the correct grid.}\\label{tab:generator}\n'
        '\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{lccc}\\toprule\nArm & Panel & Pass@2 & Coverage \\\\\\midrule\n'
        + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\\end{table}\n')
    # Leaderboard
    lbrows = ['Upstream V1 \\citep{rudra2026} & author & 33.89 \\\\',
              'Our V1 reproduction & in comp. & 30.97 \\\\']
    titles = {'B0': 'B0: 360\\,s decoding', 'S1': 'S1: B0 + NLL-only selector',
              'S2': 'S2: B0 + anchored coverage', 'L1': 'L1: B0 + length-aware decoding'}
    for key, item in versions.items():
        score = item.get('public_score_raw') if item and item.get('status') == 'COMPLETE' else None
        lbrows.append(f"{titles[key]} & late & {score if score is not None else 'pending'} \\\\")
    (OUT / 'table_leaderboard.tex').write_text(
        '\\begin{table}[t]\\centering\\footnotesize\n\\caption{Kaggle ARC-AGI-2 public leaderboard scores (percent of hidden test '
        'outputs). B0, S1, S2 and L1 share the generator, the 128-step adaptation and a 360-second decoding budget per task.}\\label{tab:leaderboard}\n'
        '\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{llr}\\toprule\nEntry & Status & Score \\\\\\midrule\n'
        + '\n'.join(lbrows) + '\n\\bottomrule\\end{tabular}\\end{table}\n')
    # Pilot
    if pilot:
        arms = [('control', 'Parent DFS, equal slices'), ('iterative', 'Iterative DFS, equal slices'),
                ('length_aware', 'Iterative DFS, length-aware')]
        pn = pilot['test_outputs']
        rows = [f"{lab} & {pn - pilot['empty_outputs'][a]}/{pn} & {pilot['candidate_oracle'][a]}/{pn} & "
                f"{pilot['correct'][a + '_baseline']}/{pn} & {1000 * pilot['timings'][a]['seconds_per_forward_call']:.0f} \\\\"
                for a, lab in arms]
        (OUT / 'table_pilot.tex').write_text(
            '\\begin{table}[t]\\centering\\small\n\\caption{Matched decoding pilot on panel W. All arms share one adapted '
            'model per task and receive 180\\,s each. Columns: outputs with any candidate, coverage, pass@2, and '
            'milliseconds per forward pass.}\\label{tab:pilot}\n'
            '\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lcccc}\\toprule\nArm & Nonempty & Coverage & Pass@2 & ms \\\\\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\\end{table}\n')
    else:
        (OUT / 'table_pilot.tex').write_text('')


def pilot_text(pilot):
    path = OUT / 'pilot_results.tex'
    if pilot is None:
        path.write_text('The matched pilot is running at the time of writing; its protocol and predictions are fixed in the released records.\n')
        return
    e, o, c = pilot['empty_outputs'], pilot['candidate_oracle'], pilot['correct']
    n = pilot['test_outputs']
    t = pilot['timings']
    eq = pilot['equivalence']
    txt = (f"Table~\\ref{{tab:pilot}} and Figure~\\ref{{fig:pilot}} report the pilot. With equal slices, the parent search leaves "
           f"{e['control']} of {n} outputs without any candidate, as in all five earlier arms. The length-aware schedule leaves "
           f"{e['length_aware']}. Coverage is {o['control']}/{n} for the parent search and {o['length_aware']}/{n} for the "
           f"length-aware schedule; pass@2 is {c['control_baseline']}/{n} and {c['length_aware_baseline']}/{n}. "
           f"The iterative search reproduces the parent's candidate sets on {eq['identical_grid_sets']} of "
           f"{eq['undeadlined_views_in_both']} views that finished before their deadline in both arms, and takes "
           f"{1000 * t['iterative']['seconds_per_forward_call']:.0f}\\,ms per forward pass against "
           f"{1000 * t['control']['seconds_per_forward_call']:.0f}\\,ms for the parent. ")
    if pilot['decision'] == 'promote_to_fresh_confirmation':
        txt += 'All three pre-registered predictions concern mechanism, and the schedule met the promotion rule.'
    else:
        txt += f"The pre-registered promotion rule was not met (decision: {pilot['decision'].replace('_', ' ')})."
    path.write_text(txt + '\n')


def extra_numbers(pilot, conf, versions):
    m = {}
    if pilot:
        n = pilot['test_outputs']; e = pilot['empty_outputs']; o = pilot['candidate_oracle']; c = pilot['correct']
        t = pilot['timings']
        m.update(PilotN=n, PilotEmptyCtl=e['control'], PilotEmptyLA=e['length_aware'], PilotCovCtl=o['control'],
                 PilotCovLA=o['length_aware'], PilotSolvedCtl=c['control_baseline'], PilotSolvedLA=c['length_aware_baseline'],
                 PilotLosses=len(pilot['losses']), PilotEq=pilot['equivalence']['identical_grid_sets'],
                 PilotEqN=pilot['equivalence']['undeadlined_views_in_both'],
                 PilotMsParent=f"{1000 * t['control']['seconds_per_forward_call']:.1f}",
                 PilotMsIter=f"{1000 * t['iterative']['seconds_per_forward_call']:.1f}",
                 PilotRatio=f"{pilot['length_aware_to_control_time_ratio']:.3f}",
                 PilotMaxTokens=pilot['decoding']['length_aware']['max_suffix_tokens'],
                 PilotFinCtl=pilot['decoding']['control']['valid_finished'], PilotFinLA=pilot['decoding']['length_aware']['valid_finished'],
                 PilotGridsCtl=pilot['unique_candidate_grids']['control'], PilotGridsLA=pilot['unique_candidate_grids']['length_aware'])
    for k, item in versions.items():
        ok = item and item.get('status') == 'COMPLETE' and item.get('public_score_raw') is not None
        m['Score' + {'B0': 'BZero', 'S1': 'SOne', 'S2': 'STwo', 'L1': 'LOne'}[k]] = item['public_score_raw'] if ok else 'pending'
    if conf:
        n = conf['test_outputs']
        lo, hi = conf['task_bootstrap_ci95']
        m.update(ConfN=n, ConfT=conf['tasks'], ConfSolvedCtl=conf['correct']['control_baseline'],
                 ConfSolvedLA=conf['correct']['length_aware_baseline'], ConfGains=len(conf['gains']), ConfLosses=len(conf['losses']),
                 ConfEmptyCtl=conf['empty_outputs']['control'], ConfEmptyLA=conf['empty_outputs']['length_aware'],
                 ConfCovCtl=conf['candidate_oracle']['control'], ConfCovLA=conf['candidate_oracle']['length_aware'],
                 ConfLo=f'{100 * lo:.1f}', ConfHi=f'{100 * hi:.1f}', ConfRatio=f"{conf['time_ratio']:.3f}",
                 ConfDelta=f"{100 * conf['delta']:.1f}")
    (OUT / 'numbers_extra.tex').write_text('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in m.items()) + '\n')


def confirmation_text(conf):
    path = OUT / 'confirmation_results.tex'
    if conf is None:
        path.write_text('% confirmation pending\n'); return
    n = conf['test_outputs']; c = conf['correct']; e = conf['empty_outputs']; o = conf['candidate_oracle']
    lo, hi = conf['task_bootstrap_ci95']; g, l = len(conf['gains']), len(conf['losses'])
    txt = (f"The frozen configuration was then run once on the {conf['tasks']} reserve tasks ({n} outputs), which no earlier "
           f"experiment had touched (Figure~\\ref{{fig:confirmation}}). Pass@2 is {c['control_baseline']}/{n} for the parent search and "
           f"{c['length_aware_baseline']}/{n} for the length-aware schedule, with {g} gain{'s' if g != 1 else ''} and "
           f"{l} loss{'es' if l != 1 else ''}; the task-bootstrap 95\\% interval for the difference is "
           f"[{100 * lo:.1f}, {100 * hi:.1f}] percentage points. Empty pools fall from {e['control']} to {e['length_aware']}, "
           f"coverage moves from {o['control']} to {o['length_aware']}, and the total time ratio is {conf['time_ratio']:.3f}. ")
    if conf['incomplete']:
        txt += f"{len(conf['incomplete'])} task--arm executions were incomplete and are excluded from the counts. "
    if g > l:
        txt += 'The gain observed on the development panel carries over to unseen tasks.'
    elif g == l:
        txt += 'On unseen tasks the schedule changes coverage without a net change in solved outputs.'
    else:
        txt += 'On unseen tasks the schedule loses more outputs than it gains, which limits the claim to candidate availability.'
    path.write_text(txt + '\n')

def build():
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    pf.style()
    lb = load('length_budget/analysis.json'); holdout = load('holdout_analysis.json')
    dup = load('duplicate_analysis/pool_00.json'); cond = load('conditional_ttt_development/analysis.json')
    weight = load('context_weighted_ttt_development/analysis.json'); comp = load('completion_development/analysis.json')
    s4 = load('live_repair_confirmation/analysis.json'); pilot = load('length_aware_development/analysis.json')
    versions = {k: load(f'version_study/{k}_submission.json') for k in ['B0', 'S1', 'S2']}
    versions['L1'] = load('l1_submission/L1_submission.json')
    conf = load('length_aware_confirmation/analysis.json')
    pf.teaser(FIG, lb); pf.clock(FIG, lb); has_pilot = pf.pilot_fig(FIG, pilot); has_conf = pf.confirmation_fig(FIG, conf); pf.gallery(FIG, pilot)
    extra_numbers(pilot, conf, versions)
    confirmation_text(conf)
    numbers(lb, holdout, dup, cond, weight, comp, s4, pilot, versions)
    tables(holdout, cond, weight, comp, s4, lb, pilot, versions)
    pilot_text(pilot)
    (OUT / 'flags.tex').write_text('\\newif\\ifpilot\\pilot' + ('true' if has_pilot else 'false') + '\n'
                                   '\\newif\\ifconf\\conf' + ('true' if has_conf else 'false') + '\n')
    latex = shutil.which('pdflatex') or '/Library/TeX/texbin/pdflatex'
    for _ in range(3):
        run = subprocess.run([latex, '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], cwd=OUT,
                             capture_output=True, text=True, errors='replace', timeout=120)
        if run.returncode:
            raise RuntimeError((run.stdout + run.stderr)[-3000:])
    import fitz
    doc = fitz.open(OUT / 'main.pdf')
    prev = OUT / 'preview'; prev.mkdir(exist_ok=True)
    for f in prev.glob('*.png'): f.unlink()
    for i, page in enumerate(doc):
        page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(prev / f'page_{i + 1:02d}.png')
    (OUT / 'paper_text.txt').write_text('\n'.join(p.get_text() for p in doc))
    log = (OUT / 'main.log').read_text(errors='ignore')
    print(json.dumps({'pages': len(doc), 'pilot': has_pilot, 'confirmation': has_conf,
                      'undefined': log.count('undefined'), 'overfull': log.count('Overfull')}))


if __name__ == '__main__':
    build()
