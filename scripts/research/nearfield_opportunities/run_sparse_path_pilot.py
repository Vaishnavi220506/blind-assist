"""Bounded sparse-path pilot: native SkyDiscover versus independent candidates.

Candidate code is parsed as a literal policy, never executed. Generation sees
Development information only. Held geometry is created after both arms seal.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'artifacts.local/work/ba-sparse-path-20260922/run-v1'
OLD = REPO / 'artifacts.local/work/ba-observation-mechanisms-20260922/run-v1'
SKY = Path('E:/SkyDiscover')
CLI = Path('E:/codex-tools/bin/codex.exe')
PROTOCOL = REPO / 'research/active/dtr-r0/nearfield/SPARSE_PATH_PROTOCOL_20260922.md'
HELD_GRID = dict(sides=[-1, 1], widths=[.09, .15, .25, .43],
                 depths=[1.10, 1.70, 2.20, 2.70], margins=[.009, .021, .033])


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core():
    spec = importlib.util.spec_from_file_location('sparse_path_frozen', ROOT / 'sparse_path_core.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    import active_view as av
    import active_view_positive as pos
    import path_observation_analysis as path
    import sparse_path_core as c
    if ROOT.exists():
        raise FileExistsError('create-once evidence root already exists')
    av.verify_seal(OLD / 'completion-seal.json')
    ROOT.mkdir(parents=True)
    for src in [Path(__file__), Path(c.__file__), Path(av.__file__), PROTOCOL]:
        shutil.copyfile(src, ROOT / src.name)
    observations = path.index_views(read(OLD / 'all-pose-observations.json'))
    pairs = read(OLD / 'pairs.json')
    pair_ids = {member: pair['id'] for pair in pairs for member in pair['members']}
    dev = [dict(id=row['id'], truth=av.intersects_query(pos.source_scene(row)),
                pair_id=pair_ids[row['id']], bins=[list(observations[row['id']][(round(i*.01, 10), 0.)]['bins']) for i in range(13)])
           for row in read(OLD / 'source.json')]
    write(ROOT / 'development.json', dev)
    baselines = c.build_baselines(dev)
    write(ROOT / 'baselines.json', baselines)
    baseline_results = {name: {str(k): c.summarize(dev, c.apply_baseline(definition, dev, k)) for k in (3, 4)} for name, definition in baselines.items()}
    write(ROOT / 'development_baselines.json', baseline_results)
    seed = {str(k): {'default': c.apply_baseline(baselines['exact_global'], dev, k)[dev[0]['id']], 'rules': []} for k in (3, 4)}
    (ROOT / 'seed.py').write_text('POLICY = ' + repr(seed) + '\n', encoding='utf-8')
    # All class targets are derived exclusively from already-consumed Development.
    groups = defaultdict(list)
    for row in dev:
        groups[tuple(row['bins'][0])].append(row)
    targets = []
    for initial, rows in sorted(groups.items()):
        targets.append(dict(initial=list(initial), count=len(rows), schedules={str(k): c.apply_baseline(baselines['exact_initial_lookup'], rows, k)[rows[0]['id']] for k in (3, 4)}))
    task = '''Improve a sparse +X path sampling POLICY. The source is synthetic 8-ray ToF, 10cm radial bins, 12cm motion. Choose all positions using ONLY the initial eight bins. Indices 0..12 mean centimetres. For budget3 use 3 positions; budget4 use4. Each includes0 and12 and is strictly increasing. Different initial bins may choose different paths; no intermediate adaptation in this pilot.
Output a complete Python code block with ONLY POLICY = <literal dictionary> (optional docstring); no functions/imports/loops or executable expressions. Schema: {'3': {'default':[0,6,12], 'rules':[{'feature':'min','value':20,'schedule':[0,5,12]}]}, '4': {'default':[0,6,8,12], 'rules':[]}}. Each budget has at most3 ordered rules. A rule matches when feature(initial)<=value; first match wins, otherwise default. Features: min, max, left_min (bins0..3), right_min (bins4..7), spread(max-min), argmin(first smallest index), near_count(number bins<=30). Threshold must be finite. Both budgets must exist. No identifiers or hidden source data are inputs.
Objective: total finite-cohort pure-signature scenes at budgets3 and4, with total designated pair separation /1000 as tiebreak. Signature includes positions and quantized measurements. This is INFORMATION, not alert classification. A signature is pure only if all Development scenes sharing it have the same truth. Candidate output must satisfy interface; invalid code scores zero. Start from strong exact-global schedules in the seed. Seek a short generalizable rule that exploits initial-range/geometric differences. Developer class-optimal schedule examples below provide a training hint, NOT hidden-evaluation data. No tools, files, network, commands, memory or agents: all input is in this prompt. First-level subagent only; do not create, spawn, invoke, contact, or delegate to another agent. Report extra work to the root agent.
'''
    task += '\nDevelopment initial-class targets:\n' + json.dumps(targets, separators=(',', ':'))
    (ROOT / 'task.txt').write_text(task, encoding='utf-8')
    evaluator = '''import importlib.util,json
from pathlib import Path
from skydiscover.optimize.evaluation.evaluation_result import EvaluationResult
ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('frozen_sparse_core',ROOT/'sparse_path_core.py')
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
def evaluate(program_path):
    try:
        policy=c.load_policy(program_path)
        rows=json.loads((ROOT/'development.json').read_text())
        summaries={str(k):c.summarize(rows,{r['id']:c.select(policy,r['bins'][0],k) for r in rows}) for k in (3,4)}
        metrics={'combined_score':c.score(summaries),'validity':1}
        for k,s in summaries.items():
            metrics['resolved_'+k]=s['resolved_scenes'];metrics['pairs_'+k]=s['separated_pairs']
        return EvaluationResult(metrics,{'feedback':json.dumps({'counts':metrics,'remaining_mixed_ids':{k:len(rows)-s['resolved_scenes'] for k,s in summaries.items()}})})
    except Exception as exc:
        return EvaluationResult({'combined_score':0,'validity':0},{'feedback':str(exc)})
'''
    (ROOT / 'evaluator.py').write_text(evaluator, encoding='utf-8')
    for arm in ('direct', 'sky'):
        folder = ROOT / arm
        folder.mkdir()
        config = dict(max_iterations=3, checkpoint_interval=1, max_parallel_iterations=1,
            language='python', diff_based_generation=False, max_solution_length=12000,
            llm=dict(models=[dict(name='codex-cli/gpt-5.6-sol',weight=1.)], reasoning_effort='medium',
                     timeout=240, retries=0, codex_executable=str(CLI)),
            search=dict(type='best_of_n' if arm=='direct' else 'topk',num_context_programs=0 if arm=='direct' else 1,
                        database=dict(random_seed=220922, **({'best_of_n':100} if arm=='direct' else {}))),
            prompt=dict(system_message=task),
            evaluator=dict(timeout=30,max_retries=0,cascade_evaluation=False,inject_evaluator_context=False),
            monitor=dict(enabled=False))
        write(folder / 'config.json', config)
    write(ROOT/'manifest.json', dict(id='ba-sparse-path-20260922',scope='consumed Development search, new same-generator held geometry',
        source_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        sky_commit=subprocess.check_output(['git','-C',str(SKY),'rev-parse','HEAD'],text=True).strip(),
        cli_path=str(CLI),cli_sha256=sha(CLI),cli_version='codex-cli 0.149.1',held_grid=HELD_GRID,
        budget=dict(calls_each=3,evals_each=4,tokens_each=120000,seconds_per_call=240),
        retry='none; uncertain dispatched calls consume slot; no resumption after process loss',
        compute='TASK_NOT_GPU_SUITABLE scalar analytic replay; local model control; no remote worker',
        files={**{p.name:sha(p) for p in ROOT.iterdir() if p.is_file()},
               **{f'{a}/config.json':sha(ROOT/a/'config.json') for a in ('direct','sky')}},
        old_files={str(p):sha(p) for p in OLD.iterdir() if p.is_file()}))
    print(json.dumps({'prepared':str(ROOT),'dev_scenes':len(dev),'baseline_counts':{name:{k:(v['resolved_scenes'],v['separated_pairs']) for k,v in result.items()} for name,result in baseline_results.items()}}),flush=True)


async def search(arm, canary=False):
    from skydiscover.optimize.config import Config
    from skydiscover.optimize.runner import Runner
    from skydiscover.optimize.llm.codex_cli import CodexCliLLM
    from skydiscover.execution_budget import BudgetLedger, BudgetCeilings
    from skydiscover.optimize.search.default_discovery_controller import DiscoveryController
    folder = ROOT / arm
    manifest = read(ROOT / 'manifest.json')
    for filename, expected in manifest['files'].items():
        assert sha(ROOT/filename)==expected, filename
    assert sha(CLI)==manifest['cli_sha256']
    terminal = folder / ('canary_receipt.json' if canary else 'terminal.json')
    dispatch = folder / ('canary_start.json' if canary else 'start.json')
    if dispatch.exists():
        raise FileExistsError('Run slot already consumed; do not restart')
    write(dispatch,dict(pid=os.getpid(),started_unix=time.time()))
    temp = folder / 'tmp'
    temp.mkdir(exist_ok=True)
    os.environ['TEMP']=os.environ['TMP']=str(temp)
    tempfile.tempdir=str(temp)
    c = core()
    started=time.monotonic()
    class Journal(BudgetLedger):
        def flush(self):
            write(folder / ('canary_budget.json' if canary else 'budget.json'),self.to_receipt())
        def start_generation(self, **kwargs):
            event=super().start_generation(**kwargs); self.flush(); return event
        def finish_generation(self, *args, **kwargs):
            try: return super().finish_generation(*args, **kwargs)
            finally: self.flush()
        def start_evaluation(self, **kwargs):
            event=super().start_evaluation(**kwargs); self.flush(); return event
    budget=Journal(BudgetCeilings(3,4,120000))
    call_count=0
    async def recorded_provider(self,prompt,timeout,response_format):
        nonlocal call_count
        call_count+=1
        call=folder/f'call_{call_count:02d}'
        call.mkdir()
        (call/'prompt.txt').write_text(prompt,encoding='utf-8')
        args=[str(CLI),'exec','--json','--ephemeral','--sandbox','read-only','--ignore-user-config',
              '--color','never','--skip-git-repo-check','--cd',str(call),'--model',self.model,
              '-c','model_reasoning_effort="medium"','-c','features.shell_tool=false','-c','features.multi_agent=false','-']
        write(call/'dispatch.json',dict(args=args,started_unix=time.time(),prompt_sha256=sha(call/'prompt.txt')))
        process=await asyncio.create_subprocess_exec(*args,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        write(call/'process.json',dict(pid=process.pid))
        t=time.monotonic()
        try:
            stdout,stderr=await asyncio.wait_for(process.communicate(prompt.encode()),timeout=timeout)
        except BaseException:
            await self._stop_process(process)
            write(call/'receipt.json',dict(status='in_doubt_no_retry',process_released=True,elapsed_s=time.monotonic()-t))
            raise
        (call/'events.jsonl').write_bytes(stdout);(call/'stderr.log').write_bytes(stderr)
        write(call/'receipt.json',dict(status='completed' if process.returncode==0 else 'failed',returncode=process.returncode,elapsed_s=time.monotonic()-t,process_released=True))
        if process.returncode: raise RuntimeError('Codex process failed')
        events=[json.loads(line) for line in stdout.decode().splitlines() if line.strip()]
        if any(e.get('type')=='item.completed' and e.get('item',{}).get('type') not in ('agent_message','reasoning') for e in events):
            raise RuntimeError('Forbidden model tool use')
        return self._parse_jsonl(stdout.decode())
    CodexCliLLM._run_once=recorded_provider
    native_discovery=DiscoveryController.run_discovery
    async def one_attempt_per_iteration(self,*args,**kwargs):
        kwargs['retry_times']=1
        return await native_discovery(self,*args,**kwargs)
    DiscoveryController.run_discovery=one_attempt_per_iteration
    DiscoveryController.skip_test_rescore=True
    config=Config.from_dict(read(folder/'config.json'))
    runner=Runner(str(ROOT/'evaluator.py'),str(ROOT/'seed.py'),config=config,output_dir=str(folder/('canary' if canary else 'search')))
    try:
        if canary:
            from skydiscover.optimize.evaluation.evaluator import Evaluator
            config.evaluator.evaluation_file=str(ROOT/'evaluator.py')
            evaluator=Evaluator(config.evaluator)
            result=await evaluator.evaluate_program((ROOT/'seed.py').read_text(),program_id='seed-canary')
            assert result.metrics['validity']==1
            write(terminal,dict(status='PASS',metrics=result.metrics,model_calls=0))
            print(json.dumps(read(terminal)),flush=True)
            return
        with budget.activate():
            best=await runner.run(iterations=3)
        if best is None: raise RuntimeError('No best candidate')
        (folder/'selected.py').write_text(best.solution,encoding='utf-8')
        c.load_policy(folder/'selected.py')
        write(terminal,dict(status='COMPLETED',elapsed_s=time.monotonic()-started,best=best.to_dict(),selected_sha256=sha(folder/'selected.py'),budget=budget.to_receipt()))
    except BaseException as exc:
        write(terminal,dict(status='FAILED_NO_RETRY',error=str(exc),elapsed_s=time.monotonic()-started,budget=budget.to_receipt()))
        raise
    finally:
        budget.flush()
    print(json.dumps({'arm':arm,'status':'COMPLETED','metrics':best.metrics,'usage':budget.to_receipt()['usage']}),flush=True)


def finish():
    import active_view as av
    c=core(); manifest=read(ROOT/'manifest.json')
    assert sha(Path(av.__file__))==manifest['files']['active_view.py']
    for filename,expected in manifest['files'].items():assert sha(ROOT/filename)==expected,filename
    if (ROOT/'held_start.json').exists(): raise FileExistsError('Held slot already consumed, never rescore')
    terminals={a:read(ROOT/a/'terminal.json') for a in ('direct','sky')}
    if any(t['status']!='COMPLETED' for t in terminals.values()): raise RuntimeError('Incomplete arms')
    selections={a:sha(ROOT/a/'selected.py') for a in terminals}
    assert all(selections[a]==terminals[a]['selected_sha256'] for a in terminals)
    with (ROOT/'held_start.json').open('x',encoding='utf-8') as stream:
        json.dump({'started_unix':time.time(),'selected':selections,'retry':'none'},stream)
    write(ROOT/'selection_seal.json',dict(selected=selections,created_unix=time.time(),held_generated=False))
    grid=manifest['held_grid']; held=[]; sources=[]
    old_keys={json.dumps(row['scene'],sort_keys=True) for row in read(OLD/'source.json')}
    for side,width,depth,margin in itertools.product(grid['sides'],grid['widths'],grid['depths'],grid['margins']):
        pair_id=f'held_pair_{len(held)//2:03d}'
        for offset in (-margin,margin):
            scene=av.Scene((av.Box(round(side*(.30+width/2+offset),10),depth,width),))
            source=asdict(scene)
            assert json.dumps(source,sort_keys=True) not in old_keys
            ident=f'held_{len(held):03d}'
            sources.append(dict(id=ident,scene=source))
            held.append(dict(id=ident,pair_id=pair_id,truth=av.intersects_query(scene),bins=[list(av.observe(scene,(round(j*.01,10),0.))) for j in range(13)]))
    write(ROOT/'held_source.json',sources);write(ROOT/'held_observations.json',held)
    dev=read(ROOT/'development.json');baselines=read(ROOT/'baselines.json')
    policies={a:c.load_policy(ROOT/a/'selected.py') for a in terminals}
    results={}
    for split,rows in [('development',dev),('held',held),('joint',dev+held)]:
        full=c.summarize(rows,{r['id']:list(range(13)) for r in rows})
        endpoint=c.summarize(rows,{r['id']:[0,12] for r in rows})
        methods={}
        for name in list(baselines)+list(policies):
            methods[name]={}
            for k in (3,4):
                schedules=({r['id']:c.select(policies[name],r['bins'][0],k) for r in rows} if name in policies else c.apply_baseline(baselines[name],rows,k))
                value=c.summarize(rows,schedules)
                assert set(endpoint['resolved_ids'])<=set(value['resolved_ids'])<=set(full['resolved_ids'])
                assert set(endpoint['separated_pair_ids'])<=set(value['separated_pair_ids'])<=set(full['separated_pair_ids'])
                value['full_resolved_retained_fraction']=value['resolved_scenes']/full['resolved_scenes'] if full['resolved_scenes'] else None
                value['lost_vs_full_ids']=sorted(set(full['resolved_ids'])-set(value['resolved_ids']))
                value['held_resolved_in_joint']=sum(i.startswith('held_') for i in value['resolved_ids']) if split=='joint' else None
                methods[name][str(k)]=value
                if split=='held':write(ROOT/f'held_choices_{name}_{k}.json',schedules)
        results[split]=dict(full=full,endpoint=endpoint,methods=methods)
    for path,expected in manifest['old_files'].items():assert sha(path)==expected,path
    write(ROOT/'result.json',dict(results=results,terminals={a:{'metrics':t['best']['metrics'],'usage':t['budget']['usage'],'elapsed_s':t['elapsed_s']} for a,t in terminals.items()},selection_seal=selections,held_scenes=len(held),scope='same-generator information only, not classifier accuracy'))
    write(ROOT/'completion_seal.json',{p.name:sha(p) for p in ROOT.iterdir() if p.is_file()})
    print(json.dumps({'held':{name:{k:(v['resolved_scenes'],v['separated_pairs']) for k,v in values.items()} for name,values in results['held']['methods'].items()},'full':(results['held']['full']['resolved_scenes'],results['held']['full']['separated_pairs'])}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','canary','search','finish'])
    parser.add_argument('--arm',choices=['direct','sky'],default='direct')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='finish':finish()
    else:asyncio.run(search(args.arm,args.action=='canary'))
