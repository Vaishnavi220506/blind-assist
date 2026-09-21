"""Replay submitted probes through SkySynth's unmodified validate_test.py."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from skysynth_audit_pilot import ROOT, SKY, load, save


def main(arms):
    gate = SKY / 'skydiscover/synthesize/workflow/scripts/validate_test.py'
    # Windows searches System32 before PATH for bare bash. Pin Git Bash only;
    # keep the official validator and all of its scoring logic byte-identical.
    runner = ROOT / 'pinned_shell_runner.py'
    runner.write_text('''import runpy, subprocess, sys
original = subprocess.Popen
def pinned(args, *a, **kw):
    if isinstance(args, (list, tuple)) and args and args[0] == 'bash':
        args = ['E:/codex-tools/tools/git/bin/bash.exe', *args[1:]]
    return original(args, *a, **kw)
subprocess.Popen = pinned
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
''', encoding='utf-8')
    env = dict(os.environ)
    env['PATH'] = 'E:/codex-tools/tools/git/bin;' + env['PATH']
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    evidence = []
    for arm in arms:
        text = (ROOT / arm / 'answer.json').read_text(encoding='utf-8').strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        findings = json.loads(text)['findings']
        suite = ROOT / arm / 'kept_test_validation'
        suite.mkdir(exist_ok=True)
        # Fixed adapter code; model content remains non-executable JSON.
        adapter = '''import json, os, sys
from pathlib import Path
sys.path.insert(0, RESEARCH_PATH)
from skysynth_audit_pilot import function, check
fn = function(Path(os.environ['SKYDISCOVER_IMPL']).read_text(encoding='utf-8'))
probes = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
sys.exit(0 if all(check(fn, p) for p in probes) else 1)
'''.replace('RESEARCH_PATH', repr(str(Path(__file__).resolve().parent)))
        (suite / 'adapter.py').write_text(adapter, encoding='utf-8')
        (suite / 'test.sh').write_text(f'#!/usr/bin/env bash\nset -eu\n"{Path(sys.executable).as_posix()}" -B adapter.py "$1"\n', encoding='utf-8', newline='\n')
        for i, finding in enumerate(findings):
            test = suite / f'finding_{i:02d}.json'
            save(test, finding['probes'])
            args = [sys.executable, '-B', str(runner), str(gate), '--test', str(test), '--reference', str(ROOT / 'reference.py')]
            for j in range(1, 9):
                args.extend(['--mutant', str(ROOT / f'M{j:02d}.py')])
            result = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, timeout=120)
            evidence.append(dict(arm=arm, finding=finding['id'], exit_code=result.returncode,
                                 output=result.stdout + result.stderr))
    save(ROOT / ('official_validate_test_' + '_'.join(arms) + '.json'), evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('direct', 'auditor'))
    args = parser.parse_args()
    main([args.arm] if args.arm else ['direct', 'auditor'])
