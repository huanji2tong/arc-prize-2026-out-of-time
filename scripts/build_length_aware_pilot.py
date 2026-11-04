"""Prepare a private three-arm length-aware decoding pilot; this script never launches GPU work.

Arms share one 128-step task-adapted model and the same 180-second budget:
  control       parent recursive DFS, equal view slices (as in all earlier pilots)
  iterative     iterative DFS from src/fast_dfs.py, same slices (equivalence and step time)
  length_aware  iterative DFS + per-output budget ledger + first-descent extension

The launch requires explicit user authorization recorded in continuation_budget.json.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_conditional_ttt import replace_once  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'paper/candidate_evidence_20261103/length_aware_development'
ARMS = ['control', 'iterative', 'length_aware']
ARM_SECONDS = 180

ARM_LOOP_OLD = '''                for batch_index, subkeys in enumerate(batches):

                    now = time.time()
                    if now >= arm_deadline - 2:
                        break
                    # Every remaining view batch receives a share of the remaining time.
                    batch_deadline = min(arm_deadline, now + (arm_deadline-now)/(len(batches)-batch_index))
                    batch_started = now
'''
ARM_LOOP_NEW = '''                ledger = None
                output_of = [sk[0].split(".")[0] for sk in batches]
                if arm_name == "length_aware":
                    # Each test output keeps an equal share of the arm budget; time saved or
                    # overrun by one output's batches is charged only to that output.
                    outputs_here = sorted(set(output_of))
                    ledger = {o: (arm_deadline - time.time()) / len(outputs_here) for o in outputs_here}
                    left = {o: output_of.count(o) for o in outputs_here}
                for batch_index, subkeys in enumerate(batches):

                    now = time.time()
                    if now >= arm_deadline - 2:
                        break
                    extend_until = None
                    if ledger is None:
                        # Every remaining view batch receives a share of the remaining time.
                        batch_deadline = min(arm_deadline, now + (arm_deadline-now)/(len(batches)-batch_index))
                    else:
                        out_key = output_of[batch_index]
                        if ledger[out_key] <= 2:
                            left[out_key] -= 1
                            batch_times.append({"batch": batch_index, "seconds": 0.0, "view_keys": subkeys,
                                                "skipped_exhausted_output_budget": True})
                            continue
                        batch_deadline = min(arm_deadline, now + ledger[out_key] / left[out_key])
                        extend_until = min(arm_deadline, now + ledger[out_key])
                    batch_started = now
'''
DFS_OLD = '                    dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, arm_max_score, batch_deadline)'
DFS_NEW = '''                    dfs_trace = {}
                    if arm_name == "control":
                        dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, arm_max_score,
                                                         batch_deadline, trace=dfs_trace)
                    else:
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
RECORD_OLD = '"view_keys": subkeys})'
RECORD_NEW = ('"view_keys": subkeys, "dfs_trace": dfs_trace,\n'
              '                                        "dfs_seconds": dfs_finished-batch_started,\n'
              '                                        "slice_seconds": batch_deadline-batch_started,\n'
              '                                        "extension_cap_seconds": None if extend_until is None else extend_until-batch_started,\n'
              '                                        "overran_slice": dfs_finished > batch_deadline + 0.5,\n'
              '                                        "raw_finished": raw_finished, "valid_finished": valid_finished,\n'
              '                                        "max_suffix_tokens": max_suffix_tokens})\n'
              '                    if ledger is not None:\n'
              '                        ledger[out_key] -= time.time() - batch_started\n'
              '                        left[out_key] -= 1')


def make_solver(source):
    source = replace_once(source, 'import json\n', 'import json\nimport hashlib\nfrom fast_dfs import inference_turbo_dfs_iterative\n')
    source = replace_once(source, '            stats = trainer.train()',
        '            stats = trainer.train()\n            assert stats.global_step == 128, "Incomplete shared task adaptation"')
    source = replace_once(source, '        puzzle_ds_multi = puzzle_ds.split_multi_replies()',
        '''        adapted_hash = hashlib.sha256()
        delta_sq = 0.
        changed_tensors = 0
        for name, tensor in sorted(get_peft_model_state_dict(model, adapter_name="default").items()):
            delta = (tensor.detach().float() - default_weights[name].float())
            delta_sq += float(delta.square().sum().item())
            changed_tensors += int(bool(torch.any(delta != 0).item()))
            adapted_hash.update(name.encode())
            adapted_hash.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        assert changed_tensors > 0 and delta_sq > 0, "Task adaptation did not change parameters"
        puzzle_ds_multi = puzzle_ds.split_multi_replies()''')
    source = replace_once(source, '"training_steps": stats.global_step, "arms": {}}',
        '"training_steps": stats.global_step, "arms": {}, "shared_adaptation": True,\n'
        '                       "adapted_weights_sha256": adapted_hash.hexdigest(),\n'
        '                       "adapter_delta_l2": delta_sq ** .5, "changed_adapter_tensors": changed_tensors}')
    # Telemetry on the parent recursive DFS (unchanged search).
    source = replace_once(source,
        'def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time) -> dict:',
        'def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time, trace=None) -> dict:')
    source = replace_once(source, '    n = logits.size(0)',
        '    if trace is not None: trace["prefix_calls"] = trace.get("prefix_calls", 0) + 1\n    n = logits.size(0)')
    source = replace_once(source, '            score = nll[i, t].item()',
        '            score = nll[i, t].item()\n            if trace is not None and score >= max_score:\n'
        '                trace["pruned_extensions"] = trace.get("pruned_extensions", 0) + 1')
    source = replace_once(source, '                if t == EOS_ID:',
        '                if t == EOS_ID:\n                    if trace is not None: trace["eos_within_cutoff"] = trace.get("eos_within_cutoff", 0) + 1')
    source = source.replace('        end_time=end_time,', '        end_time=end_time,\n        trace=trace,')
    source = replace_once(source, '    return suffixes',
        '    if trace is not None and time.time() >= end_time: trace["deadline_hit"] = True\n    return suffixes')
    source = replace_once(source,
        'def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):',
        'def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time, trace=None):')
    # Three arms, all at the parent cutoff; order rotates over all six permutations.
    source = replace_once(source, '''        arm_names = ["control", "wide"]
        if stable_seed(key) % 2:
            arm_names.reverse()''', '''        import itertools
        arm_names = list(list(itertools.permutations(["control", "iterative", "length_aware"]))[stable_seed(key) % 6])''')
    source = replace_once(source, 'arm_max_score = -np.log({"control": 0.2, "wide": 0.05}[arm_name])',
                          'arm_max_score = -np.log(0.2)')
    source = source.replace('arm_start + 360', f'arm_start + {ARM_SECONDS}')
    source = source.replace('"requested_seconds": 360', f'"requested_seconds": {ARM_SECONDS}')
    source = source.replace('arm_start+360', f'arm_start+{ARM_SECONDS}')
    source = replace_once(source, ARM_LOOP_OLD, ARM_LOOP_NEW)
    source = replace_once(source, DFS_OLD, DFS_NEW)
    source = replace_once(source, RECORD_OLD, RECORD_NEW)
    assert '"wide"' not in source
    ast.parse(source)
    return source


def main():
    BASE.mkdir(exist_ok=True)
    if (BASE / 'run.json').exists():
        assert json.loads((BASE / 'run.json').read_text())['status'] == 'PREPARED', 'Accepted runs are immutable'
    ids = json.loads((BASE.parent / 'context_weighted_ttt_development/protocol.json').read_text())['tasks']
    registry = json.loads((BASE.parent / 'task_usage_registry.json').read_text())
    assert set(ids) <= set(registry['development_tasks'])
    fast = (ROOT / 'src/fast_dfs.py').read_text()
    analysis = json.loads((BASE.parent / 'length_budget/analysis.json').read_text())
    protocol = {
        'tasks': ids, 'test_outputs': 9, 'scope': 'Reused eight-task development panel (7/9 outputs empty in all five earlier arms)',
        'arms': {'control': 'parent recursive DFS, equal remaining-time slices',
                 'iterative': 'iterative DFS (src/fast_dfs.py), equal slices; same search order as parent',
                 'length_aware': 'iterative DFS, per-output ledger, first-descent extension'},
        'seconds_per_arm': ARM_SECONDS, 'cutoff_probability': 0.2,
        'prediction_before_run': {
            'mechanism': 'Equal slices of 45 s (22.5 s for two-output tasks) admit about 430 (215) forward passes at the '
                         f"measured {analysis['timing']['median_seconds_per_call']:.4f} s per pass; the seven empty outputs need 506-930 tokens.",
            'falsifiable': ['length_aware empty valid pools below control (7/9 in all prior arms)',
                            'iterative candidates identical to control on batches where neither search hits a deadline',
                            'iterative seconds per forward pass not above control']},
        'primary_metrics': ['empty valid pools', 'candidate oracle', 'pass@2'],
        'primary_comparison': 'length_aware minus control, paired by output',
        'secondary': 'iterative minus control step time; candidate equality on undeadlined batches',
        'stop_rule': 'Promote to a fresh-task confirmation only if length_aware reduces empty pools without losing any control solve and time per arm is comparable',
        'novelty': 'Iterative DFS is an implementation change; deadline extension is a scheduling rule; no new model claim',
        'competition_submission': False, 'reserve_tasks_consumed': 0,
        'budget': {'notebook_wall_seconds': 3600, 'max_quota_hours': 1.5},
        'launch_authorization': 'User on 2026-11-04 13:15 (+08:00) asked to pursue the paper prize by any method; one run, at most 1.5 GPU quota hours, keeping at least 4 hours',
        'fast_dfs_sha256': hashlib.sha256(fast.encode()).hexdigest()}
    solver = make_solver((ROOT / 'src/paired_runtime/arc_solver.py').read_text())
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
            s = replace_once(s, 'time.time() + 4800 - 300', 'time.time() + 3600 - 180')
        elif s.startswith('from arc_loader import ArcDataset'):
            s = replace_once(s, 'selected_ids = ordered[4:16]', 'selected_ids = ' + repr(ids))
            s = replace_once(s, 'len(selected_ids) == 12', 'len(selected_ids) == 8')
            s = s.replace('"arm_seconds": 360', f'"arm_seconds": {ARM_SECONDS}')
            s = s.replace('"notebook_timeout_seconds": 4800', '"notebook_timeout_seconds": 3600')
            s = s.replace('"arms": {"control": 0.2, "wide": 0.05}', '"arms": {"control": 0.2, "iterative": 0.2, "length_aware": 0.2}')
            s += '\nprotocol = ' + repr(protocol) + '\n(artifacts/"protocol.json").write_text(json.dumps(protocol,indent=2))\n'
        s = s.replace('for arm in ["control", "wide"]', 'for arm in ' + repr(ARMS).replace("'", '"'))
        s = s.replace('six submission schemas valid', 'nine submission schemas valid')
        s = s.replace('Twelve-task development pilot;', 'Eight-task development pilot;')
        s = s.replace('all six submissions', 'all nine submissions')
        assert '"wide"' not in s, s[:200]
        c['source'] = s.splitlines(True)
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None
    solver_cell = next(i for i, c in enumerate(nb['cells']) if ''.join(c['source']).startswith('%%writefile arc_solver.py'))
    nb['cells'].insert(solver_cell, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                                     'source': ('%%writefile fast_dfs.py\n' + fast).splitlines(True)})
    nb['cells'][0]['source'] = ['# ARC2 length-aware decoding pilot\n',
                                'One shared adapted model per task; parent DFS, iterative DFS, and iterative DFS with a '
                                'per-output ledger and first-descent extension. Eight reused development tasks. Private research only.\n']
    for i, c in enumerate(nb['cells']):
        c['id'] = f'length-aware-{i}'
        if c['cell_type'] == 'code':
            s = ''.join(c['source'])
            ast.parse(s.split('\n', 1)[1] if s.startswith('%%writefile') else s)
    joined = '\n'.join(''.join(c['source']) for c in nb['cells'])
    for needle in ['%%writefile fast_dfs.py', 'from fast_dfs import inference_turbo_dfs_iterative',
                   'for arm in ["control", "iterative", "length_aware"]', 'selected_ids = ' + repr(ids)]:
        assert needle in joined, needle
    folder = ROOT / 'kaggle_api/length_aware_pilot'
    folder.mkdir(exist_ok=True)
    target = folder / 'arc2-length-aware-decoding-pilot-v1.ipynb'
    target.write_text(json.dumps(nb, indent=1))
    meta = json.loads((ROOT / 'kaggle_api/paired_search/kernel-metadata.json').read_text())
    meta.update(id='huangjt66/arc2-length-aware-decoding-pilot-v1', title='ARC2 Length Aware Decoding Pilot V1',
                code_file=target.name)
    assert meta.get('is_private', True) in (True, 'true') and not meta.get('enable_internet', False)
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    (BASE / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    receipt = {'status': 'PREPARED', 'launch_authorized': True, 'notebook': str(target), 'folder': str(folder),
               'source_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
               'protocol_sha256': hashlib.sha256((BASE / 'protocol.json').read_bytes()).hexdigest(),
               'kernel_ref': meta['id'], 'api_timeout_seconds': 3600, 'max_quota_hours': 1.5,
               'expected_run_directory': str(BASE / 'gpu_run')}
    (BASE / 'run.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
