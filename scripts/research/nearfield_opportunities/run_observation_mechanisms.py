"""One frozen cohort for five observation mechanisms; no hidden-source policy."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import active_view as av
import active_view_positive as positive
import boundary_separability as boundary
import two_step_observation as two

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT/"artifacts.local/work/ba-active-view-20260921/run-v1"
TRANSFER = ROOT/"artifacts.local/work/ba-opportunity-transfer-20260921/run-v1"
BOUNDARY = ROOT/"artifacts.local/work/ba-boundary-separability-20260922/run-v1"
DOCS = ROOT/"research/active/dtr-r0/nearfield"
PROTOCOL = DOCS/"OBSERVATION_MECHANISMS_PROTOCOL_20260922.md"
CONFIG = dict(sides=[-1,1], widths=[.07,.19,.37], depths=[.95,1.45,1.95,2.45,2.95],
              margins=[.007,.017,.027], thickness=.04, wall_z=4.2)
POLICIES = ("fixed_endpoint", "positive_endpoint", "fixed_two_step", "lookahead_two_step")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_source():
    scenes, pairs = [], []
    for side,width,depth,margin in itertools.product(CONFIG["sides"],CONFIG["widths"],CONFIG["depths"],CONFIG["margins"]):
        members = []
        for offset in (-margin,margin):
            ident = f"scene_{len(scenes):03d}"
            scene = av.Scene((av.Box(round(side*(.30+width/2+offset),10),depth,width),))
            scenes.append(dict(id=ident,scene=asdict(scene)))
            members.append(ident)
        pairs.append(dict(id=f"pair_{len(pairs):03d}",members=members,side=side,width=width,depth=depth,margin=margin))
    return pairs, scenes


def diagnostic_poses():
    poses = {(0.,0.)}
    for x,z in av.ACTIONS:
        poses.update((round(x*n/12,8),round(z*n/12,8)) for n in range(1,13))
    poses.update(itertools.product((-.06,.06),repeat=2))
    return tuple(sorted(poses))


def metric_contrast(rows, candidate, baseline):
    result = {}
    for name,state,truth in (("TP","INTERSECTS",True),("FP","INTERSECTS",False),
                             ("false_OUT","NONINTERSECTING_HYPOTHESES",True),("correct_negative","NONINTERSECTING_HYPOTHESES",False)):
        a = {r["id"] for r in rows if r["truth"]==truth and r["policies"][candidate]["decision"]==state}
        b = {r["id"] for r in rows if r["truth"]==truth and r["policies"][baseline]["decision"]==state}
        result[name] = dict(gained=sorted(a-b),lost=sorted(b-a),retained=len(a&b))
    return result


def run(output):
    # Optional solver/scientific modules must exist before the single run starts.
    import continuous_boundary_witness as witness
    import path_observation_analysis as path_analysis
    if output.exists():
        raise FileExistsError("Refusing to overwrite scored evidence")
    output.mkdir(parents=True)
    start, stages = time.perf_counter(), []
    def stage(name):
        row=dict(stage=name,seconds=time.perf_counter()-start)
        stages.append(row)
        print(json.dumps(row),flush=True)
    try:
        for folder in (OLD,TRANSFER,BOUNDARY):
            av.verify_seal(folder/"completion-seal.json")
        old_hashes = {str(p.resolve()):av.file_hash(p) for folder in (OLD,TRANSFER,BOUNDARY) for p in folder.iterdir() if p.is_file()}
        paths = [Path(__file__),Path(__file__).with_name("test_two_step_observation.py"),Path(two.__file__),
            Path(av.__file__),Path(positive.__file__),Path(boundary.__file__),Path(witness.__file__),
            Path(witness.__file__).with_name("test_continuous_boundary_witness.py"),Path(path_analysis.__file__),
            Path(path_analysis.__file__).with_name("test_path_observation_analysis.py"),PROTOCOL,
            DOCS/"CONTINUOUS_BOUNDARY_WITNESS_BRIEF_20260922.md",DOCS/"PATH_OBSERVATION_BRIEF_20260922.md"]
        for path in paths:
            shutil.copyfile(path,output/path.name)
        pairs,scenes = make_source()
        old_keys = {boundary.geometry_key(row["scene"]) for p in (OLD/"source.json",TRANSFER/"active-source.json",BOUNDARY/"source.json") for row in read(p)}
        keys = [boundary.geometry_key(r["scene"]) for r in scenes]
        assert len(pairs)==90 and len(keys)==len(set(keys))==180 and not old_keys.intersection(keys)
        av.write_json(output/"source.json",scenes)
        av.write_json(output/"pairs.json",pairs)
        av.write_json(output/"config.json",dict(**CONFIG,poses=diagnostic_poses(),actions=av.ACTIONS,
            steps=two.STEPS,query=av.QUERY,range_step=av.RANGE_STEP,python=sys.executable,backend="CPU / TASK_NOT_GPU_SUITABLE"))
        av.write_json(output/"admission.json",dict(pairs=90,scenes=180,new_geometry_overlap=0,old_file_hashes=old_hashes))
        av.seal(output/"source-seal.json",{p.name:p for p in output.iterdir() if p.is_file()})
        stage("all_source_code_and_briefs_frozen_before_observation")
        # Simulator-only cache: never passed to policy or witness solver.
        source_by_id = {r["id"]:r for r in scenes}
        cache = {}
        def observe(ident,pose):
            key=(ident,tuple(pose))
            if key not in cache:
                cache[key]=dict(camera=list(pose),**boundary.observe_with_raw(positive.source_scene(source_by_id[ident]),tuple(pose)))
            return cache[key]
        initial = [dict(id=s["id"],bins=observe(s["id"],av.ORIGIN)["bins"]) for s in scenes]
        av.write_json(output/"initial-observations.json",initial)
        hypotheses=av.hypothesis_scenes()
        labels=tuple(av.intersects_query(s) for s in hypotheses)
        forecasts=two.public_forecasts(hypotheses)
        original=read(OLD/"public-model-forecasts.json")
        bank=av.ForecastBank(tuple(map(tuple,original["initial"])),tuple(tuple(map(tuple,a)) for a in original["forecasts"]),tuple(original["labels"]))
        assert forecasts[av.ORIGIN]==bank.initial and labels==bank.labels
        assert all(forecasts[a]==bank.forecasts[i] for i,a in enumerate(av.ACTIONS))
        av.write_json(output/"tree-public-forecasts.json",dict(labels=labels,poses=[dict(camera=p,bins=f) for p,f in forecasts.items()]))
        first=[]
        for r in initial:
            plan=two.first_choice(r["bins"],forecasts,labels)
            endpoint=positive.select_positive_priority(tuple(r["bins"]),bank)
            first.append(dict(id=r["id"],lookahead=plan,positive_endpoint=endpoint["action_index"],fixed_first=0))
        av.write_json(output/"first-choices.json",first)
        av.seal(output/"first-choice-seal.json",dict(choices=output/"first-choices.json",initial=output/"initial-observations.json",
            prior=output/"tree-public-forecasts.json",source=output/"source-seal.json"))
        stage("all_first_and_endpoint_actions_sealed_before_actual_future_views")
        av.verify_seal(output/"first-choice-seal.json")
        first=read(output/"first-choices.json")
        mids=[]
        for r in first:
            mids.append(dict(id=r["id"],fixed=observe(r["id"],two.STEPS[0]),lookahead=observe(r["id"],two.STEPS[r["lookahead"]["action"]])))
        av.write_json(output/"selected-mid-observations.json",mids)
        seconds=[]
        for r,m in zip(first,mids,strict=True):
            assert r["id"]==m["id"]
            pose=tuple(m["lookahead"]["camera"])
            candidates=two.update(r["lookahead"]["candidates"],m["lookahead"]["bins"],pose,forecasts)
            second=two.second_choice(candidates,pose,forecasts,labels)
            seconds.append(dict(id=r["id"],lookahead=second,mid_candidates=list(candidates),
                lookahead_final_pose=two.add(pose,two.STEPS[second["action"]]),fixed_final_pose=av.ACTIONS[0]))
        av.write_json(output/"second-choices.json",seconds)
        av.seal(output/"second-choice-seal.json",dict(choices=output/"second-choices.json",mid=output/"selected-mid-observations.json",
            first=output/"first-choice-seal.json",prior=output/"tree-public-forecasts.json"))
        stage("all_second_actions_sealed_after_mid_before_final_observations")
        av.verify_seal(output/"second-choice-seal.json")
        seconds=read(output/"second-choices.json")
        finals=[]
        predictions=[]
        for init,first_row,mid,second in zip(initial,first,mids,seconds,strict=True):
            ident=init["id"]
            assert ident==first_row["id"]==mid["id"]==second["id"]
            endpoint=observe(ident,av.ACTIONS[0])
            positive_obs=observe(ident,av.ACTIONS[first_row["positive_endpoint"]])
            chosen=observe(ident,tuple(second["lookahead_final_pose"]))
            finals.append(dict(id=ident,fixed_endpoint=endpoint,positive_endpoint=positive_obs,lookahead=chosen))
            candidates=tuple(first_row["lookahead"]["candidates"])
            after_mid=two.update(candidates,mid["fixed"]["bins"],two.STEPS[0],forecasts)
            remain={"fixed_endpoint":two.update(candidates,endpoint["bins"],av.ACTIONS[0],forecasts),
                "positive_endpoint":two.update(candidates,positive_obs["bins"],tuple(positive_obs["camera"]),forecasts),
                "fixed_two_step":two.update(after_mid,endpoint["bins"],av.ACTIONS[0],forecasts),
                "lookahead_two_step":two.update(second["mid_candidates"],chosen["bins"],tuple(chosen["camera"]),forecasts)}
            predictions.append(dict(id=ident,policies={p:dict(decision=two.decision(v,labels),remaining=list(v),remaining_count=len(v)) for p,v in remain.items()}))
        av.write_json(output/"selected-final-observations.json",finals)
        av.write_json(output/"policy-predictions.json",predictions)
        av.seal(output/"policy-prediction-seal.json",{p.name:p for p in output.iterdir() if p.is_file()})
        stage("policy_predictions_sealed_before_evaluator_and_diagnostic_views")
        observations=[dict(id=s["id"],views=[observe(s["id"],p) for p in diagnostic_poses()]) for s in scenes]
        assert len(cache)==180*53
        av.write_json(output/"all-pose-observations.json",observations)
        av.seal(output/"all-observation-seal.json",dict(observations=output/"all-pose-observations.json",policy=output/"policy-prediction-seal.json"))
        stage("all_9540_evaluator_views_sealed")
        # Distinct public signatures share exactly the same solver result.
        queries={}
        query_ids=[]
        for s in scenes:
            row=dict(id=s["id"],queries={})
            for mode,poses in (("initial",[av.ORIGIN]),("fixed_endpoint",[av.ORIGIN,av.ACTIONS[0]])):
                obs=[dict(camera=list(p),bins=observe(s["id"],p)["bins"]) for p in poses]
                key=av.canonical(obs).decode()
                if key not in queries:
                    queries[key]=dict(query_id=f"query_{len(queries):03d}",observations=obs)
                row["queries"][mode]=queries[key]["query_id"]
            query_ids.append(row)
        av.write_json(output/"witness-public-queries.json",list(queries.values()))
        av.write_json(output/"witness-query-mapping.json",query_ids)
        av.seal(output/"witness-input-seal.json",dict(queries=output/"witness-public-queries.json",mapping=output/"witness-query-mapping.json",
            observations=output/"all-observation-seal.json",solver=output/Path(witness.__file__).name,brief=output/"CONTINUOUS_BOUNDARY_WITNESS_BRIEF_20260922.md"))
        answers=[]
        for i,q in enumerate(queries.values()):
            answer=dict(**q,result=witness.infer(q["observations"]))
            answers.append(answer)
            av.write_json(output/f"witness-{q['query_id']}.json",answer)
            if i%10==0 or i==len(queries)-1:
                print(json.dumps(dict(witness_queries_done=i+1,total=len(queries),seconds=time.perf_counter()-start)),flush=True)
        av.write_json(output/"witness-predictions.json",answers)
        av.seal(output/"witness-prediction-seal.json",{p.name:p for p in output.iterdir() if p.is_file()})
        stage("all_continuous_witnesses_sealed_before_actual_truth_evaluation")
        av.verify_seal(output/"witness-prediction-seal.json")
        truth={r["id"]:av.intersects_query(positive.source_scene(r)) for r in scenes}
        rows=[dict(**p,truth=truth[p["id"]]) for p in predictions]
        by_query={a["query_id"]:a for a in answers}
        by_prediction={r["id"]:r for r in rows}
        initial_by_id={r["id"]:tuple(r["bins"]) for r in initial}
        witness_rows=[]
        validated_witnesses=0
        for a in answers:
            for label,geometry in a["result"]["witnesses"].items():
                if geometry is None:
                    continue
                scene=positive.source_scene(dict(scene=geometry))
                assert av.intersects_query(scene)==(label=="IN")
                assert all(av.observe(scene,tuple(o["camera"]))==tuple(o["bins"]) for o in a["observations"])
                validated_witnesses+=1
        for r in query_ids:
            for mode,qid in r["queries"].items():
                answer=by_query[qid]["result"]
                if mode=="initial":
                    support=tuple(i for i,bins in enumerate(bank.initial) if bins==initial_by_id[r["id"]])
                    old_decision=two.decision(support,labels)
                else:
                    support=by_prediction[r["id"]]["policies"]["fixed_endpoint"]["remaining"]
                    old_decision=by_prediction[r["id"]]["policies"]["fixed_endpoint"]["decision"]
                opposite="OUT" if old_decision=="INTERSECTS" else "IN" if old_decision=="NONINTERSECTING_HYPOTHESES" else None
                witness_rows.append(dict(id=r["id"],mode=mode,query_id=qid,truth=truth[r["id"]],
                    status=answer["status"],reason=answer["reason"],valid_witness_count=answer["valid_witness_count"],
                    true_label_witness_found=answer["witnesses"]["IN" if truth[r["id"]] else "OUT"] is not None,
                    original_bank_no_match=not support,original_bank_decision=old_decision,
                    opposing_witness_to_original_commit=opposite is not None and answer["witnesses"][opposite] is not None))
        witness_summary={mode:dict(cases=len(part),reasons=dict(Counter(v["reason"] for v in part)),
            witness_counts=dict(Counter(str(v["valid_witness_count"]) for v in part)),
            true_label_witness_found=sum(v["true_label_witness_found"] for v in part),
            original_bank_no_match=sum(v["original_bank_no_match"] for v in part),
            no_bank_match_with_constructive_witness=sum(v["original_bank_no_match"] and v["valid_witness_count"]>0 for v in part),
            opposing_witness_to_original_commit=sum(v["opposing_witness_to_original_commit"] for v in part),
            commitments=0,UNKNOWN=len(part)) for mode in ("initial","fixed_endpoint")
            for part in ([v for v in witness_rows if v["mode"]==mode],)}
        av.write_json(output/"witness-evaluation.json",witness_rows)
        metrics={p:positive.metrics(rows,p) for p in POLICIES}
        for p in POLICIES:
            metrics[p]["measurements_per_case"]=3 if p.endswith("two_step") else 2
        contrast=metric_contrast(rows,"lookahead_two_step","fixed_two_step")
        gate=dict(adds_TP=metrics["lookahead_two_step"]["tp"]>metrics["fixed_two_step"]["tp"],
            retains_fixed_TP=not contrast["TP"]["lost"],no_new_wrong=not contrast["FP"]["gained"] and not contrast["false_OUT"]["gained"],
            correct_decisive_not_reduced=metrics["lookahead_two_step"]["correct_decisive"]>=metrics["fixed_two_step"]["correct_decisive"])
        evaluator_obs={r["id"]:{tuple(v["camera"]):v["bins"] for v in r["views"]} for r in observations}
        path_result=path_analysis.analyze(pairs,scenes,observations)
        av.write_json(output/"path-analysis.json",path_result)
        av.write_json(output/"policy-evaluation.json",rows)
        summary=dict(two_step=dict(metrics=metrics,contrasts={b:metric_contrast(rows,"lookahead_two_step",b) for b in POLICIES if b!="lookahead_two_step"},
            gate=gate,decision="PUBLIC_TWO_STEP_GAIN" if all(gate.values()) else "PUBLIC_TWO_STEP_GATE_NOT_MET",
            evaluator_ceiling=two.evaluator_ceiling(evaluator_obs,truth)),
            witness_unique_queries=len(queries),witness_statuses=dict(Counter(a["result"]["status"] for a in answers)),
            witness_evaluation=witness_summary,independently_replayed_witnesses=validated_witnesses,
            costs=dict(evaluator_views=9540,path_views=13,path_motion_m=.12,two_step_views=3,two_step_motion_m=.12,endpoint_views=2,endpoint_motion_m=.12))
        av.write_json(output/"summary.json",summary)
        assert all(av.file_hash(Path(p))==h for p,h in old_hashes.items())
        stage("all_accounting_complete_old_evidence_unchanged")
        av.write_json(output/"stage-order.json",stages)
        av.write_json(output/"local-disposition.json",dict(terminal_id="ba-observation-mechanisms-20260922",role="COMPONENT_OR_CHALLENGER",mode="COMPONENT",
            scope="Five fixed analytic mechanism checks; per-mechanism outcomes retain their evidence boundaries",two_step=summary["two_step"]["decision"],
            global_registration="PENDING_SUPPORTED_COMMAND",persistent_resources=[]))
        av.seal(output/"completion-seal.json",{p.name:p for p in output.iterdir() if p.is_file()})
        print(json.dumps(dict(status="COMPLETED",seconds=stages[-1]["seconds"],witness_queries=len(queries),two_step=summary["two_step"]["decision"])),flush=True)
    except BaseException as exc:
        av.write_json(output/"failure.json",dict(type=type(exc).__name__,message=str(exc),stages=stages))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    canonical=(ROOT/"artifacts.local/work/ba-observation-mechanisms-20260922").resolve()
    if canonical not in args.output.resolve().parents:
        parser.error("Output must be a new child of canonical mechanism artifact root")
    run(args.output)
