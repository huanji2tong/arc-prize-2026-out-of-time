"""Build the L1 late-submission notebook: B0 matched control + length-aware decoding.

Only the decoder and the view-batch schedule change; the model, 128-step adaptation,
360-second per-task decoding budget, cutoff, rescoring, and occurrence selector are
identical to B0.  This script never pushes or submits.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conditional_ttt import replace_once  # noqa: E402
from build_length_aware_pilot import ARM_LOOP_OLD, ARM_LOOP_NEW, DFS_OLD, RECORD_OLD, RECORD_NEW  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'paper/candidate_evidence_20261103/l1_submission'
VERSION = {'key': 'L1', 'slug': 'arc2-paper-l1-length-aware', 'title': 'ARC2 Paper L1 Length Aware Decoding',
           'selector': 'baseline', 'role': 'Length-aware decoding with the B0 generator, budget, and selector',
           'question': 'Does length-aware decoding change the hidden-set score relative to the matched B0 control?'}
DFS_L1 = '''                    dfs_trace = {}
                    dfs_result = inference_turbo_dfs_iterative(
                        model, tokens, max_new_tokens, arm_max_score, batch_deadline,
                        arc_tokens=ARC_TOKENS, eos_id=EOS_ID, pad_id=PAD_ID, trace=dfs_trace,
                        extend_until=extend_until)
                    dfs_finished = time.time()
                    raw_finished = sum(len(beams) for _, beams in dfs_result)
                    valid_finished = sum(formatter.convert_tokens_to_array(ys) is not None
                                         for _, beams in dfs_result for _, ys in beams)
                    max_suffix_tokens = max([len(ys) for _, beams in dfs_result for _, ys in beams] or [0])
'''


def patch_solver(source):
    source = replace_once(source, 'import json\n', 'import json\nfrom fast_dfs import inference_turbo_dfs_iterative\n')
    loop = ARM_LOOP_NEW.replace('if arm_name == "length_aware":', 'if True:  # L1: every arm uses the ledger')
    source = replace_once(source, ARM_LOOP_OLD, loop)
    source = replace_once(source, DFS_OLD, DFS_L1)
    source = replace_once(source, RECORD_OLD, RECORD_NEW)
    ast.parse(source)
    return source


def build():
    STUDY.mkdir(exist_ok=True)
    protocol = {'objective': 'Linked Kaggle entry for the paper method: length-aware decoding under the B0 budget',
                'authorization': 'User on 2026-11-04 13:15 (+08:00) asked to pursue the ARC Prize paper award by any method',
                'max_competition_submissions': 1, 'max_normal_gpu_hours': 0.5, 'max_normal_gpu_hours_each': .5,
                'normal_internal_seconds': 1800, 'hidden_internal_seconds': 43200, 'api_timeout_seconds': 43200,
                'stop_after': '2026-11-09T20:00:00+08:00',
                'source_policy': 'Private notebooks, internet disabled, upstream attribution retained',
                'overlap_public_checks_with_scoring': True,
                'submission_policy': 'One technically validated fixed version; persist POST intent and reconcile ambiguous responses',
                'score_interpretation': 'Post-deadline score; compare descriptively with B0 under identical budget',
                'versions': [VERSION]}
    pp = STUDY / 'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol
    else:
        pp.write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    receipt_path = STUDY / 'L1_run.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text())['status'] != 'PREPARED':
        print('Keep existing L1 intent'); return
    nb = json.loads((ROOT / 'kaggle_api/paper_versions/B0/arc2-paper-b0-control.ipynb').read_text())
    fast = (ROOT / 'src/fast_dfs.py').read_text()
    for c in nb['cells']:
        s = ''.join(c['source'])
        if s.startswith('%%writefile arc_solver.py'):
            s = '%%writefile arc_solver.py\n' + patch_solver(s.split('\n', 1)[1])
        elif s.startswith('import hashlib, importlib.util'):
            s = replace_once(s, "PAPER_VARIANT='B0'", "PAPER_VARIANT='L1'")
            assert "PAPER_SELECTOR='baseline'" in s
        c['source'] = s.splitlines(True)
    solver_cell = next(i for i, c in enumerate(nb['cells']) if ''.join(c['source']).startswith('%%writefile arc_solver.py'))
    nb['cells'].insert(solver_cell, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                                     'source': ('%%writefile fast_dfs.py\n' + fast).splitlines(True)})
    nb['cells'][0]['source'] = [f"# {VERSION['title']}\n", VERSION['question'] + '\n',
                                'Identical to B0 except for the iterative DFS (exact reimplementation of the parent search) '
                                'and the length-aware view schedule (per-output budget ledger and first-descent extension). '
                                'Offline code submission. Normal execution checks four public tasks; competition rerun processes every hidden task. '
                                'Upstream Qwen/NVARC runtime attribution: Apache-2.0. New study code: MIT-0.\n']
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            s = ''.join(c['source'])
            ast.parse(s.split('\n', 1)[1] if s.startswith('%%writefile') else s)
            c['outputs'] = []; c['execution_count'] = None
    joined = '\n'.join(''.join(c['source']) for c in nb['cells'])
    for needle in ['%%writefile fast_dfs.py', 'extend_until=extend_until', "PAPER_VARIANT='L1'", 'ledger[out_key] -= time.time() - batch_started']:
        assert needle in joined, needle
    folder = ROOT / 'kaggle_api/paper_versions/L1'; folder.mkdir(parents=True, exist_ok=True)
    name = VERSION['slug'] + '.ipynb'; path = folder / name
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    meta = json.loads((ROOT / 'kaggle_api/paper_versions/B0/kernel-metadata.json').read_text())
    meta.update(id='huangjt66/' + VERSION['slug'], title=VERSION['title'], code_file=name)
    assert meta['is_private'] and not meta['enable_internet']
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    receipt = {'status': 'PREPARED', **VERSION, 'kernel_ref': meta['id'], 'folder': str(folder), 'notebook': str(path),
               'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'api_timeout_seconds': 43200,
               'max_normal_quota_hours': .5, 'expected_run_directory': str(STUDY / 'L1_normal')}
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'notebook': str(path), 'sha256': receipt['source_sha256']}))


if __name__ == '__main__':
    build()
