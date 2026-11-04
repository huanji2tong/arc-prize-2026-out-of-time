"""Build a paired weight-training experiment without modifying prior runs."""
from pathlib import Path
import ast
import hashlib
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'paper/candidate_evidence_20261103/conditional_ttt_development'
ARMS = ['all_outputs', 'after_first', 'final_output']


def replace_once(source, old, new):
    assert source.count(old) == 1, (source.count(old), old[:100])
    return source.replace(old, new, 1)


def make_solver(source):
    source = replace_once(source, 'import json\n',
        'import json\nimport hashlib\nfrom pathlib import Path\n'
        'from transformers import set_seed, TrainerCallback\n'
        'from conditional_ttt import final_completion_labels, after_first_labels\n')
    source = replace_once(source, '        return batch\n', '''        objective = getattr(self, "objective", "all_outputs")
        if objective != "all_outputs":
            mask = {"after_first": after_first_labels, "final_output": final_completion_labels}[objective]
            for i in range(len(examples)):
                masked = mask(batch["input_ids"][i].tolist(), batch["labels"][i].tolist())
                batch["labels"][i] = torch.tensor(masked, device=batch["labels"].device)
        return batch
''')
    prefix = source[:source.index('    while not queue.empty():')]
    train = source[source.index('        load_result = set_peft_model_state_dict('):
                   source.index('        memory_allocated = torch.cuda.max_memory_allocated()')]
    train = replace_once(train, '        model = FastLanguageModel.for_training(model)', '''        reset_state = get_peft_model_state_dict(model, adapter_name="default")
        assert reset_state.keys() == default_weights.keys()
        assert all(torch.equal(v, default_weights[k]) for k, v in reset_state.items()), "Adapter reset failed"
        del reset_state
        model = FastLanguageModel.for_training(model)''')
    train = replace_once(train, '        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):', '''        samples = train_ds.as_list(formatter)
        assert len(samples) == 128
        samples_hash = hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest()
        collator.objective = arm_name
        parent_objective = collator.objective
        collator.objective = "all_outputs"
        input_tokens = supervised_tokens = final_tokens = conditional_tokens = 0
        for sample in samples:
            ids = tokenizer.encode(sample["text"])
            labeled = collator.torch_call([{"input_ids": ids}])["labels"][0].tolist()
            final = final_completion_labels(ids, labeled)
            input_tokens += len(ids)
            supervised_tokens += sum(x != -100 for x in labeled)
            final_tokens += sum(x != -100 for x in final)
            conditional_tokens += sum(x != -100 for x in after_first_labels(ids, labeled))
        collator.objective = parent_objective
        training_deadline = min(time.time() + 480, end_time - 10)
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):''')
    train = replace_once(train, 'train_dataset=Dataset.from_list(train_ds.as_list(formatter)),',
                         'train_dataset=Dataset.from_list(samples),')
    train = replace_once(train, 'args=UnslothTrainingArguments(**train_args),',
        'args=UnslothTrainingArguments(**train_args),\n                callbacks=[TrainingDeadline(training_deadline)],')
    # Same generator and score_kgmon selection; only supervised positions change.
    views = source[source.index('        puzzle_ds_multi = puzzle_ds.split_multi_replies()'):
                   source.index('        task_record = {"task_id": key')]
    decode = source[source.index('            arm_start = time.time()'):
                    source.index('        telemetry_dir = ')]
    decode = decode.replace('360', '180')
    decode = replace_once(decode, '-np.log({"control": 0.2, "wide": 0.05}[arm_name])', '-np.log(0.2)')
    body = '''    class TrainingDeadline(TrainerCallback):
        def __init__(self, deadline): self.deadline = deadline
        def on_step_end(self, args, state, control, **kwargs):
            if time.time() >= self.deadline:
                control.should_training_stop = True
            return control

    assert not rerun_mode, "Private development experiment only"
    telemetry_dir = Path("/kaggle/working/arc_artifacts/task_telemetry")
    telemetry_dir.mkdir(exist_ok=True)
    while True:
        key = queue.get()
        if key is None: break
        task_record = {"task_id": key, "arms": {}}
        from itertools import permutations
        panel = sorted(json.load(open(os.environ["ARC_TASK_IDS_FILE"])))
        arm_names = list(list(permutations(["all_outputs", "after_first", "final_output"]))[panel.index(key) % 6])
        task_record["arm_order"] = arm_names
        for arm_name in arm_names:
            start_time = time.time()
            if start_time >= end_time - 30:
                task_record["arms"][arm_name] = {"incomplete": True, "reason": "global_deadline"}
                continue
            set_seed(42)
            print(f"TRAIN_START {key} {arm_name}", flush=True)
            torch.cuda.reset_peak_memory_stats()
'''
    body += textwrap.indent(train, '    ')
    body += '''            trained_state = get_peft_model_state_dict(model, adapter_name="default")
            delta_sq = 0.0
            changed_tensors = 0
            trained_hash = hashlib.sha256()
            for name, value in sorted(trained_state.items()):
                diff = value.float() - default_weights[name].float()
                sq = float(torch.sum(diff * diff).item())
                delta_sq += sq
                changed_tensors += int(sq > 0)
                trained_hash.update(name.encode())
                trained_hash.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            del trained_state, diff
            training = {"steps": stats.global_step, "seconds": time.time() - start_time,
                        "trainer_seconds": stats.metrics.get("train_runtime"),
                        "reported_loss": stats.training_loss, "samples_sha256": samples_hash,
                        "input_tokens": input_tokens,
                        "supervised_tokens": {"all_outputs": supervised_tokens, "after_first": conditional_tokens, "final_output": final_tokens}[arm_name],
                        "adapter_delta_l2": delta_sq ** .5, "changed_adapter_tensors": changed_tensors,
                        "trained_adapter_sha256": trained_hash.hexdigest(), "reset_verified": True}
            print(f"TRAIN_DONE {key} {arm_name} steps={stats.global_step}", flush=True)
'''
    body += textwrap.indent(views, '    ') + decode
    body += '''            task_record["arms"][arm_name]["training"] = training
            task_record["arms"][arm_name]["total_seconds"] = time.time() - start_time
            task_record["arms"][arm_name]["incomplete"] = stats.global_step != 128 or task_record["arms"][arm_name]["global_budget_truncated"]
            (telemetry_dir / (key + ".json")).write_text(json.dumps(task_record, indent=2))
        (telemetry_dir / (key + ".json")).write_text(json.dumps(task_record, indent=2))
'''
    return prefix + body


def main():
    BASE.mkdir(exist_ok=True)
    receipt_path = BASE / 'run.json'
    if receipt_path.exists():
        assert json.loads(receipt_path.read_text())['status'] == 'PREPARED', 'Accepted source is immutable'
    source_nb = ROOT / 'notebooks/arc2_paired_search_pilot_v1.ipynb'
    assert hashlib.sha256(source_nb.read_bytes()).hexdigest() == 'e45cf1bd7bd97fe968119796adbea12b9041c2c16357678e8f0b96c75e60756e'
    nb = json.loads(source_nb.read_text())
    solver = make_solver((ROOT / 'src/paired_runtime/arc_solver.py').read_text())
    ast.parse(solver)
    runtime = BASE / 'runtime'; runtime.mkdir(exist_ok=True)
    (runtime / 'arc_solver.py').write_text(solver)
    challenges_path = Path('/Users/tom/.cache/arc2026/data/arc-agi_evaluation_challenges.json')
    challenges = json.loads(challenges_path.read_text())
    ordered = sorted(challenges, key=lambda k: hashlib.sha256(('arc-neural-v1:' + k).encode()).hexdigest())
    panel = [k for k in ordered[4:16] if len(challenges[k]['train']) >= 3]
    def train_size(k):
        return sum(len(row) for pair in challenges[k]['train'] for grid in pair.values() for row in grid)
    ids = sorted(panel, key=lambda k: (train_size(k), k))[:8]
    registry = json.loads((BASE.parent / 'task_usage_registry.json').read_text())
    assert set(ids) <= set(registry['development_tasks'])
    protocol = {
        'question': 'Does restricting LoRA supervision to the final demonstration improve conditional grid prediction?',
        'task_selection': 'From prior ordered[4:16], require >=3 demonstrations; smallest 8 by demonstration cell count, then task ID',
        'tasks': ids, 'test_outputs': sum(len(challenges[k]['test']) for k in ids),
        'scope': 'Reused 8-task development panel with at least 3 demonstrations; no untouched confirmation claim',
        'primary_comparison': 'final_output minus after_first exact-match pass@2',
        'secondary_comparison': 'both conditional objectives versus all_outputs',
        'known_method_control': 'PoE (Franzen et al., ICML 2025), section 5.2 excludes first demonstration output from loss',
        'arms': ARMS, 'only_intervention': 'Mask direct loss on earlier assistant completions',
        'training': {'steps': 128, 'learning_rate': 5e-5, 'seed': 42, 'augmentation_seed': 1,
                     'same_input_sequences': True, 'reset_adapter_each_arm': True,
                     'max_seconds_per_arm': 480, 'lora_rank': 256, 'lora_alpha': 32},
        'inference': {'seconds_per_arm': 180, 'sequence_probability_floor': .2,
                      'views': 16, 'selector': 'unchanged score_kgmon'},
        'budget': {'notebook_wall_seconds': 3600, 'max_quota_hours': 2.0},
        'confirmation': 'Only consider one fixed confirmation on reserved tasks if complete paired development improves pass@2 without >10% time imbalance',
        'statistics': {'unit': 'task', 'bootstrap_replicates': 5000, 'seed': 20261103},
        'training_validation_note': 'Rotating targets are training examples, not independent demonstration holdouts',
        'novelty': 'Conditional supervision and task adaptation are known ideas; this experiment establishes no priority claim',
        'source_data_sha256': hashlib.sha256(challenges_path.read_bytes()).hexdigest(),
        'objective_sha256': hashlib.sha256((ROOT / 'src/conditional_ttt.py').read_bytes()).hexdigest(),
        'runtime_sha256': hashlib.sha256(solver.encode()).hexdigest(),
    }
    (BASE / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    for c in nb['cells']:
        s = ''.join(c.get('source', []))
        if s.startswith('%%writefile arc_solver.py'):
            s = '%%writefile arc_solver.py\n' + solver
        elif s.startswith('import hashlib, importlib.util'):
            s = replace_once(s, 'time.time() + 4800 - 300', 'time.time() + 3600 - 180')
            s += '\nassert not rerun_mode, "Weight-training development trial only"\n'
        elif s.startswith('from arc_loader import ArcDataset'):
            s = s.replace('selected_ids = ordered[4:16]', 'selected_ids = ' + repr(ids))
            s = s.replace('assert len(selected_ids) == 12', 'assert len(selected_ids) == 8')
            s = s.replace('["control", "wide"]', repr(ARMS))
            s = s.replace('{"control": 0.2, "wide": 0.05}', '{"all_outputs": 0.2, "after_first": 0.2, "final_output": 0.2}')
            s = s.replace('"arm_seconds": 360', '"arm_seconds": 180')
            s = s.replace('"notebook_timeout_seconds": 4800', '"notebook_timeout_seconds": 3600')
            s = s.replace('shared task-specific model, 128 augmented samples; arm order counterbalanced by task hash',
                          'independent reset adapters; same 128 samples; only loss mask differs; six arm orders rotated by sorted task index')
            s += '\nprotocol = ' + repr(protocol) + '\n(artifacts/"protocol.json").write_text(json.dumps(protocol, indent=2))\n'
        elif s.startswith('import subprocess\ncommand'):
            s = s.replace('"done!"', '"done!", "TRAIN_START", "TRAIN_DONE"')
        elif s.startswith('import gzip'):
            s = s.replace('["control", "wide"]', repr(ARMS))
            s = s.replace('submissions["control_baseline"]', 'submissions["all_outputs_baseline"]')
            s = s.replace('manifest["completed_utc"]', '''manifest["all_training_complete"] = all(not t["arms"].get(a, {}).get("incomplete", True) for t in telemetry for a in ["all_outputs", "after_first", "final_output"])
manifest["inference_return_code"] = return_code
manifest["completed_utc"]''')
        elif s.startswith('# Labels enter'):
            s = s.replace('Eight-task development pilot;', 'Paired weight-training development pilot;')
        if c['cell_type'] == 'code':
            c['outputs'] = []; c['execution_count'] = None
        c['source'] = s.splitlines(True)
    nb['cells'].insert(10, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
        'source': ('%%writefile conditional_ttt.py\n' + (ROOT / 'src/conditional_ttt.py').read_text()).splitlines(True)})
    nb['cells'][0]['source'] = ['# Conditional supervision for task-specific LoRA\n',
        'Three-arm paired 128-step adaptation on 8 reused development tasks. Identical training strings, parent weights, optimizer and decoding. '
        'Controls supervise every output or all outputs after the first (PoE section 5.2); candidate supervises only the final output. Earlier outputs stay in context. '
        'Upstream Qwen/NVARC solver: Koushik Rudra, Apache-2.0. New experiment code: MIT-0.\n']
    for i, c in enumerate(nb['cells']):
        c['id'] = f'conditional-ttt-{i}'
        if c['cell_type'] == 'code':
            s = ''.join(c['source']); ast.parse(s.split('\n', 1)[1] if s.startswith('%%writefile') else s)
    folder = ROOT / 'kaggle_api/conditional_ttt'; folder.mkdir(exist_ok=True)
    path = folder / 'arc2-conditional-ttt-v1.ipynb'
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    meta = json.loads((ROOT / 'kaggle_api/paired_search/kernel-metadata.json').read_text())
    meta.update(id='huangjt66/arc2-conditional-ttt-v1', title='ARC2 Conditional TTT V1', code_file=path.name)
    meta['is_private'] = True; meta['enable_internet'] = False
    meta.pop('id_no', None)
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    receipt = {'status': 'PREPARED', 'folder': str(folder), 'notebook': str(path), 'kernel_ref': meta['id'],
               'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'protocol_sha256': hashlib.sha256((BASE / 'protocol.json').read_bytes()).hexdigest(),
               'api_timeout_seconds': 3600, 'max_quota_hours': 2.0,
               'expected_run_directory': str(BASE / 'gpu_run')}
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
