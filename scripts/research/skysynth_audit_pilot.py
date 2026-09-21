"""One-shot, prompt-only Auditor ablation; never a full SkySynth run.

All model output is declarative JSON, not executable code. Mutants and the
reference remain evaluator-owned. No project implementation is modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
SKY = Path('E:/SkyDiscover')
CLI = Path('E:/codex-tools/bin/codex.exe')
SOURCE = REPO / 'research/active/dtr-r0/nearfield/ba_camera_corridor_metrics.py'
ROOT = REPO / 'artifacts.local/evidence/skysynth-audit-20260922'
MUTATIONS = [
    ('M01', 'known = [row for row in rows if row["truth"] is not None]', 'known = list(rows)'),
    ('M02', 'elif not pred["unknown"]:', 'else:'),
    ('M03', ' or row["frame_in_clip"] != index', ''),
    ('M04', ' or not isclose(row["time_s"], index * dt_s, abs_tol=1e-8)', ''),
    ('M05', 'first = next((r for r in event if r["predictions"][arm]["alert"]), None)',
     'first = next((r for r in clip[max(0, start - 1):end] if r["predictions"][arm]["alert"]), None)'),
    ('M06', 'sum(s["end_frame"] - s["start_frame"] + 1 for s in metrics["false_alert_segments"])',
     'sum(s["end_frame"] - s["start_frame"] for s in metrics["false_alert_segments"])'),
    ('M07', 'for start, end in _runs(clip, lambda r: r["truth"] is True):',
     'for start, end in _runs(clip, lambda r: r["truth"] is not False):'),
    ('M08', 'strata = {"all_known": known,',
     'strata = {"all_known": [r for r in known if not r["boundary"]],'),
]

CONTRACT = '''Audit evaluate_rows(rows, dt_s=0.2, arms=("audit",)) against this contract.
Rows carry clip_id, frame_in_clip, time_s, truth (true/false/null), boundary,
and predictions.audit with boolean alert, unknown, ambiguous. Unknown prediction
and alert may coexist: alert is still scored, unknown is separately counted.
Within each clip indices must be integer, contiguous from zero; timestamps must
equal index*dt_s (absolute tolerance 1e-8). Bad indices or timestamps raise ValueError.
Rows may arrive in any order. Clips are separate. dt_s must be finite and positive.
Unknown truth contributes only to coverage/unknown counters, never TP/FP/FN/TN.
For known truth: an alert is TP or FP; no alert on true is FN; no alert on false is
TN only if prediction unknown=false. Unknown no-alert is separately abstained.
all_known includes BOTH boundary and interior. These two strata partition it.
An event is a maximal consecutive TRUE run within a clip; false/null split runs.
It is detected only by an alert inside that run. Delay is first such alert time
minus event entry time. No detection means null times. A pre-entry alert alone
does not detect the event. Consecutive false-truth alerts define false segments;
all other frames split them. Sampled durations INCLUDE every sampled frame:
number_of_frames * dt_s, including one-frame segments. Empty ratios are null.
No undocumented requirement, style, throughput or security speculation is a finding.
Do not fix code. Produce concrete small probes with independently expected outputs.
Output at most 12 findings, each with at most 2 probes. Each probe rows is a list
of [clip_id, frame_in_clip, time_s, truth, boundary, alert, unknown, ambiguous].
Use expectation="ValueError" and assertions=[] for invalid input; otherwise
expectation="return" and assertions=[{"path":"arms.audit.event_count","value":1}].
Paths are dot-delimited result keys, with integer list indices allowed. Values
are scalar JSON values. Every probe has dt_s, rows, expectation, assertions.
Findings have id, property, explanation, probes. Return {"findings":[...]} only.
This is a bounded prompt-only comparison: do not use tools, filesystem, network,
memory, skills discovery, subprocesses, or other agents. All inputs are below.
First-level subagent only; do not create, spawn, invoke, contact, or delegate to
another agent. Report extra work to the root agent. The evaluator executes all
your probes after submission, without feedback/retry. No hidden answers provided.
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def function(code):
    namespace = {}
    exec(compile(code, '<frozen-metrics>', 'exec'), namespace)
    return namespace['evaluate_rows']


def expand(rows):
    return [dict(clip_id=r[0], frame_in_clip=r[1], time_s=r[2], truth=r[3],
                 boundary=r[4], predictions={'audit': dict(alert=r[5], unknown=r[6], ambiguous=r[7])})
            for r in rows]


def check(fn, probe):
    try:
        result = fn(expand(probe['rows']), dt_s=probe['dt_s'], arms=('audit',))
    except ValueError:
        return probe['expectation'] == 'ValueError'
    except Exception:
        return False
    if probe['expectation'] != 'return' or not probe['assertions']:
        return False
    try:
        for assertion in probe['assertions']:
            value = result
            for key in assertion['path'].split('.'):
                value = value[int(key)] if isinstance(value, list) else value[key]
            expected = assertion['value']
            if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
                if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9):
                    return False
            elif type(value) is not type(expected) or value != expected:
                return False
        return True
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def probe(rows, path=None, value=None):
    return dict(rows=rows, dt_s=0.2, expectation='return' if path else 'ValueError',
                assertions=[dict(path=path, value=value)] if path else [])


def prepare():
    if ROOT.exists():
        raise RuntimeError('create-once run exists; do not reset or overwrite')
    ROOT.mkdir(parents=True)
    reference = SOURCE.read_text(encoding='utf-8')
    (ROOT / 'reference.py').write_text(reference, encoding='utf-8')
    candidate = reference
    for name, old, new in MUTATIONS:
        assert reference.count(old) == 1, name
        (ROOT / f'{name}.py').write_text(reference.replace(old, new), encoding='utf-8')
        candidate = candidate.replace(old, new)
    (ROOT / 'candidate.py').write_text(candidate, encoding='utf-8')
    r = lambda i, truth, alert=False, unknown=False, boundary=False: ['x', i, i * .2, truth, boundary, alert, unknown, False]
    seeds = [
        probe([r(0, None)], 'known_truth_frames', 0),
        probe([r(0, False, unknown=True)], 'arms.audit.frames.all_known.TN', 0),
        probe([['x', 2, 0., False, False, False, False, False]]),
        probe([['x', 0, 3., False, False, False, False, False]]),
        probe([r(0, False, True), r(1, True)], 'arms.audit.detected_events', 0),
        probe([r(0, False, True)], 'arms.audit.false_alert_sampled_duration_s', .2),
        probe([r(0, True), r(1, None), r(2, True)], 'arms.audit.event_count', 2),
        probe([r(0, False, True, boundary=True)], 'arms.audit.frames.all_known.FP', 1),
    ]
    ref = function(reference)
    for (name, _, _), test in zip(MUTATIONS, seeds):
        assert check(ref, test), f'reference fails {name}'
        assert not check(function((ROOT / f'{name}.py').read_text()), test), f'live mutant {name}'
    save(ROOT / 'private_seed_checks.json', seeds)
    brief = (SKY / 'skydiscover/synthesize/workflow/agents/2-synthesis-loop/auditor.md').read_text(encoding='utf-8')
    # Same facts, same submission interface; only the audit instruction differs.
    instructions = {
        'direct': 'Review this Python metrics implementation for correctness bugs. Find violations of the supplied contract and provide reproducible regression probes. Be precise and avoid false positives.',
        'auditor': 'Apply the official SkySynth Auditor brief below as a bounded prompt-only audit component. The common experiment contract overrides tool use, delegation, logging and repeated iterations. Empty cold knowledge base; no external failure history is supplied to either arm. This is not full SkySynth.\n\n' + brief,
    }
    hashes = {}
    for arm, instruction in instructions.items():
        work = ROOT / arm
        work.mkdir()
        prompt = CONTRACT + '\nAUDIT INSTRUCTIONS\n' + instruction + '\nSOURCE\n```python\n' + candidate + '\n```\n'
        (work / 'prompt.txt').write_text(prompt, encoding='utf-8')
        hashes[arm] = digest(work / 'prompt.txt')
    manifest = dict(schema='skysynth-audit-pilot-v1', phase='EXPLORE engineering synthetic mutation audit',
        scope='official auditor prompt ablation only; not full SkySynth or production-defect discovery',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        source=str(SOURCE), source_sha256=digest(SOURCE), provider=str(CLI), provider_sha256=digest(CLI),
        provider_version='codex-cli 0.149.1', authentication='ChatGPT preflight passed',
        model='gpt-5.6-sol', reasoning_effort='medium', order=['direct', 'auditor'],
        budget=dict(calls_per_arm=1, wall_seconds_per_arm=360, tool_calls=0, max_findings=12, probes_per_finding=2),
        recovery='No retry/resume of dispatched calls. Interrupted dispatch is in_doubt and consumes its only slot. Completed receipts are immutable.',
        selection='Retain auditor as promising only with >=2 additional distinct seeded faults and no extra invalid findings. Tie favors direct. Not generalizable from one synthetic component.',
        metric='Valid submitted tests passing reference and failing candidate; distinct single-fault mutants killed; invalid findings and duplicate coverage reported separately.',
        source_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        skydiscover_commit=subprocess.check_output(['git', '-C', str(SKY), 'rev-parse', 'HEAD'], text=True).strip(),
        official_brief_sha256=hashlib.sha256(brief.encode()).hexdigest(), prompt_sha256=hashes,
        seeds=8, seed_checks='8/8 reference pass and 8/8 individual mutant failures before dispatch',
        compute='TASK_NOT_GPU_SUITABLE: scalar metrics and subprocess orchestration',
        files={p.name: digest(p) for p in ROOT.glob('*.py')})
    save(ROOT / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


def run(arm):
    manifest = load(ROOT / 'manifest.json')
    work = ROOT / arm
    if (work / 'dispatch.json').exists():
        raise RuntimeError('slot already consumed; inspect existing receipt, never retry')
    assert digest(CLI) == manifest['provider_sha256']
    assert digest(work / 'prompt.txt') == manifest['prompt_sha256'][arm]
    args = [str(CLI), 'exec', '--ignore-user-config', '--ephemeral', '--skip-git-repo-check',
            '--sandbox', 'read-only', '--json', '--color', 'never', '-m', manifest['model'],
            '-c', 'model_reasoning_effort="medium"', '-c', 'features.shell_tool=false',
            '-c', 'features.multi_agent=false', '-C', str(work),
            '-o', str(work / 'answer.json'), '-']
    save(work / 'dispatch.json', dict(arm=arm, started_unix=time.time(), args=args,
                                     prompt_sha256=digest(work / 'prompt.txt')))
    started = time.monotonic()
    with (work / 'events.jsonl').open('w', encoding='utf-8') as out, (work / 'stderr.log').open('w', encoding='utf-8') as err:
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True,
                                encoding='utf-8', shell=False, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        save(work / 'process.json', dict(pid=proc.pid, started_unix=time.time()))
        try:
            proc.communicate((work / 'prompt.txt').read_text(encoding='utf-8'), timeout=360)
            status = 'completed' if proc.returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            status = 'in_doubt_timeout_no_retry'
    save(work / 'receipt.json', dict(arm=arm, status=status, returncode=proc.returncode,
         elapsed_s=time.monotonic() - started, ended_unix=time.time(), process_released=proc.poll() is not None,
         events_sha256=digest(work / 'events.jsonl'), answer_sha256=digest(work / 'answer.json') if (work / 'answer.json').exists() else None))
    print((work / 'receipt.json').read_text())


def score():
    manifest = load(ROOT / 'manifest.json')
    for name, sha in manifest['files'].items():
        assert digest(ROOT / name) == sha, name
    ref = function((ROOT / 'reference.py').read_text())
    candidate = function((ROOT / 'candidate.py').read_text())
    mutants = {name: function((ROOT / f'{name}.py').read_text()) for name, _, _ in MUTATIONS}
    results = {}
    for arm in manifest['order']:
        work = ROOT / arm
        receipt = load(work / 'receipt.json')
        events = [json.loads(line) for line in (work / 'events.jsonl').read_text().splitlines() if line.strip()]
        usage = [event.get('usage') for event in events if event.get('type') == 'turn.completed']
        items = [event.get('item', {}) for event in events if event.get('type') == 'item.completed']
        tool_items = [item for item in items if item.get('type') not in ('agent_message', 'reasoning')]
        text = (work / 'answer.json').read_text().strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        answer = json.loads(text)
        assert len(answer['findings']) <= 12
        findings, killed = [], set()
        for finding in answer['findings']:
            probes = finding['probes']
            assert 0 < len(probes) <= 2
            valid = all(check(ref, p) for p in probes)
            fails_candidate = any(not check(candidate, p) for p in probes)
            kills = sorted(name for name, fn in mutants.items() if valid and any(not check(fn, p) for p in probes))
            accepted = valid and fails_candidate
            if accepted:
                killed.update(kills)
            findings.append(dict(id=finding['id'], property=finding['property'], reference_pass=valid,
                                 candidate_fails=fails_candidate, accepted=accepted, killed_mutants=kills))
        results[arm] = dict(receipt=receipt, usage=usage, tool_items=tool_items, findings=findings,
            submitted=len(findings), valid_findings=sum(f['accepted'] for f in findings),
            invalid_findings=sum(not f['accepted'] for f in findings), killed=sorted(killed), distinct_killed=len(killed),
            evidence_valid=receipt['status']=='completed' and not tool_items and len(usage)==1)
    gain = results['auditor']['distinct_killed'] - results['direct']['distinct_killed']
    results['decision'] = 'PROMISING_COMPONENT' if gain >= 2 and results['auditor']['invalid_findings'] <= results['direct']['invalid_findings'] else 'INCREMENTAL_VALUE_NOT_ESTABLISHED'
    if not all(results[a]['evidence_valid'] for a in manifest['order']):
        results['decision'] = 'NOT_EVALUABLE'
    save(ROOT / 'result.json', results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'score'])
    parser.add_argument('--arm', choices=['direct', 'auditor'])
    args = parser.parse_args()
    {'prepare': prepare, 'run': lambda: run(args.arm), 'score': score}[args.action]()
