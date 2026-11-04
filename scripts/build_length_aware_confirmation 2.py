"""Prepare the fresh-task confirmation of length-aware decoding (never pushes by itself).

Frozen configuration from the promoted pilot: one shared 128-step adapter per task,
180 s per arm, cutoff 0.2. Two arms only: parent recursive DFS with equal slices
(control) versus iterative DFS with ledger and first-descent extension (length_aware).
Tasks: the 32 reserve tasks (48 outputs) never used by this project.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conditional_ttt import replace_once  # noqa: E402
import build_length_aware_pilot as pilot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'paper/candidate_evidence_20261103/length_aware_confirmation'
ARMS = ['control', 'length_aware']
WALL = 10800


def main():
    BASE.mkdir(exist_ok=True)
    if (BASE / 'run.json').exists():
        assert json.loads((BASE / 'run.json').read_text())['status'] == 'PREPARED', 'Accepted runs are immutable'
    registry = json.loads((BASE.parent / 'task_usage_registry.json').read_text())
    ids = list(registry['reserved_for_neural_confirmation'])
    assert len(ids) == 32
    used = set(registry['development_tasks']) | set(registry['selection_confirmation_48']) | set(registry['repair_confirmation_24'])
    assert not set(ids) & used
    fast = (ROOT / 'src/fast_dfs.py').read_text()
    pilot_result = json.loads((BASE.parent / 'length_aware_development/analysis.json').read_text())
    assert pilot_result['decision'] == 'promote_to_fresh_confirmation'
    protocol = {
        'tasks': ids, 'test_outputs': registry['reserve_outputs'],
        'scope': 'Fresh confirmation on the 32 reserve tasks; first use of these tasks',
        'arms': {'control': 'parent recursive DFS, equal remaining-time slices',
                 'length_aware': 'iterative DFS, per-output ledger, first-descent extension'},
        'seconds_per_arm': 180, 'cutoff_probability': 0.2,
        'frozen_from': 'length_aware_development (pilot decision promote_to_fresh_confirmation)',
        'primary_comparison': 'length_aware minus control pass@2, paired by output, task bootstrap',
        'secondary': ['empty valid pools', 'candidate oracle', 'gains and losses', 'time per forward pass'],
        'stop_rule': 'Report whatever is observed; no further tuning on these tasks',
        'competition_submission': False, 'reserve_tasks_consumed': 32,
        'budget': {'notebook_wall_seconds': WALL, 'max_quota_hours': 2.0},
        'launch_authorization': 'User on 2026-11-04 13:15 and 19:22 (+08:00): pursue the paper prize by any method and continue',
        'fast_dfs_sha256': hashlib.sha256(fast.encode()).hexdigest()}
    solver = pilot.make_solver((ROOT / 'src/paired_runtime/arc_solver.py').read_text())
    solver = replace_once(solver,
        'arm_names = list(list(itertools.permutations(["control", "iterative", "length_aware"]))[stable_seed(key) % 6])',
        'arm_names = ["control", "length_aware"] if stable_seed(key) % 2 == 0 else ["length_aware", "control"]')
    ast.parse(solver)
    (BASE / 'runtime').mkdir(exist_ok=True)
    (BASE / 'runtime/arc_solver.py').write_text(solver)
    (BASE / 'runtime/fast_dfs.py').write_text(fast)
    protocol['runtime_sha256'] = hashlib.sha256(solver.encode()).hexdigest()
    nb = json.loads((ROOT / 'notebooks/arc2_paired_search_pilot_v1.ipynb').read_text())
    for c in nb['cells']:
        s = ''.join(c.get('source', []))
        if s.startswith('%%writefile arc_solver.py'):
            s = '%%writefile arc_solver.py\n' + solver
        elif s.startswith('import hashlib, importlib.util'):
            s = replace_once(s, 'time.time() + 4800 - 300', f'time.time() + {WALL} - 300')
        elif s.startswith('from arc_loader import ArcDataset'):
            s = replace_once(s, 'selected_ids = ordered[4:16]', 'selected_ids = ' + repr(ids))
            s = replace_once(s, 'assert len(selected_ids) == 12 and set(selected_ids) <= set(ordered[:len(ordered)//2])',
                             'assert len(selected_ids) == 32 and set(selected_ids) <= set(ordered)')
            s = s.replace('"split": "development_pilot_ordered_4_to_15"', '"split": "reserve_confirmation_32"')
            s = s.replace('"arm_seconds": 360', '"arm_seconds": 180')
            s = s.replace('"notebook_timeout_seconds": 4800', f'"notebook_timeout_seconds": {WALL}')
            s = s.replace('"arms": {"control": 0.2, "wide": 0.05}', '"arms": {"control": 0.2, "length_aware": 0.2}')
            s += '\nprotocol = ' + repr(protocol) + '\n(artifacts/"protocol.json").write_text(json.dumps(protocol,indent=2))\n'
        s = s.replace('for arm in ["control", "wide"]', 'for arm in ["control", "length_aware"]')
        s = s.replace('six submission schemas valid', 'six submission schemas valid (two arms)')
        s = s.replace('Twelve-task development pilot;', 'Thirty-two-task reserve confirmation;')
        assert '"wide"' not in s, s[:200]
        c['source'] = s.splitlines(True)
        if c['cell_type'] == 'code':
            c['outputs'] = []; c['execution_count'] = None
    solver_cell = next(i for i, c in enumerate(nb['cells']) if ''.join(c['source']).startswith('%%writefile arc_solver.py'))
    nb['cells'].insert(solver_cell, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                                     'source': ('%%writefile fast_dfs.py\n' + fast).splitlines(True)})
    nb['cells'][0]['source'] = ['# ARC2 length-aware decoding: reserve confirmation\n',
                                'Frozen configuration from the promoted pilot. 32 reserve tasks, two arms, one shared adapter per task. Private research only.\n']
    for i, c in enumerate(nb['cells']):
        c['id'] = f'la-confirm-{i}'
        if c['cell_type'] == 'code':
            s = ''.join(c['source'])
            ast.parse(s.split('\n', 1)[1] if s.startswith('%%writefile') else s)
    joined = '\n'.join(''.join(c['source']) for c in nb['cells'])
    for needle in ['%%writefile fast_dfs.py', 'from fast_dfs import inference_turbo_dfs_iterative',
                   'for arm in ["control", "length_aware"]', 'selected_ids = ' + repr(ids), f'time.time() + {WALL} - 300']:
        assert needle in joined, needle
    folder = ROOT / 'kaggle_api/length_aware_confirmation'; folder.mkdir(exist_ok=True)
    target = folder / 'arc2-length-aware-confirmation-v1.ipynb'
    target.write_text(json.dumps(nb, indent=1))
    meta = json.loads((ROOT / 'kaggle_api/paired_search/kernel-metadata.json').read_text())
    meta.update(id='huangjt66/arc2-length-aware-confirmation-v1', title='ARC2 Length Aware Confirmation V1', code_file=target.name)
    assert meta['is_private'] and not meta['enable_internet']
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    (BASE / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    receipt = {'status': 'PREPARED', 'launch_authorized': True, 'notebook': str(target), 'folder': str(folder),
               'source_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
               'protocol_sha256': hashlib.sha256((BASE / 'protocol.json').read_bytes()).hexdigest(),
               'kernel_ref': meta['id'], 'api_timeout_seconds': WALL, 'max_quota_hours': 2.0,
               'expected_run_directory': str(BASE / 'gpu_run')}
    (BASE / 'run.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
