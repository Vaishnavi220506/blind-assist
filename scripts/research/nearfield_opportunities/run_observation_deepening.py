"""Frozen consumed-trace deepening: continuous paths, witness guidance, open loop."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import time

import active_view as av
import active_view_positive as oldpos
import continuous_boundary_witness as oldwitness
import two_step_observation as two
import two_step_openloop as openloop
import witness_guided_path as guide

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/"artifacts.local/work/ba-observation-mechanisms-20260922/run-v1"
DOCS=ROOT/"research/active/dtr-r0/nearfield"
MODES=("endpoint","fixed_path","guided_path","feedback","openloop")
MAP={"IN_MODEL_CONDITIONAL":"INTERSECTS","OUT_MODEL_CONDITIONAL":"NONINTERSECTING_HYPOTHESES","UNKNOWN":"UNKNOWN"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def counts(rows,key):
    cells=Counter((r["truth"],r[key]) for r in rows)
    tp,fp=cells[True,"INTERSECTS"],cells[False,"INTERSECTS"]
    fo,tn=cells[True,"NONINTERSECTING_HYPOTHESES"],cells[False,"NONINTERSECTING_HYPOTHESES"]
    up,un=cells[True,"UNKNOWN"],cells[False,"UNKNOWN"]
    assert tp+fp+fo+tn+up+un==len(rows)
    p,n=tp+fo+up,fp+tn+un
    return dict(cases=len(rows),positive=p,negative=n,TP=tp,FP=fp,false_OUT=fo,correct_negative=tn,
                UNKNOWN_positive=up,UNKNOWN_negative=un,UNKNOWN=up+un,FN=fo+up,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/p if p else None,FPR=fp/n if n else None,
                correct_decisive=tp+tn,wrong_decisive=fp+fo)


def contrast(rows,candidate,baseline):
    result={}
    for name,state,truth in (("TP","INTERSECTS",True),("FP","INTERSECTS",False),
                             ("false_OUT","NONINTERSECTING_HYPOTHESES",True),("correct_negative","NONINTERSECTING_HYPOTHESES",False)):
        new={r["id"] for r in rows if r["truth"]==truth and r[candidate]==state}
        old={r["id"] for r in rows if r["truth"]==truth and r[baseline]==state}
        result[name]=dict(gained=sorted(new-old),lost=sorted(old-new),retained=len(new&old))
    return result


def run(output):
    import path_constraint_inference as model
    if output.exists():
        raise FileExistsError("Refusing to overwrite evidence")
    output.mkdir(parents=True)
    start,stages=time.perf_counter(),[]
    def stage(name):
        stages.append(dict(stage=name,seconds=time.perf_counter()-start))
        print(json.dumps(stages[-1]),flush=True)
    try:
        av.verify_seal(OLD/"completion-seal.json")
        old_hashes={str(p.resolve()):av.file_hash(p) for p in OLD.iterdir() if p.is_file()}
        av.write_json(output/"consumed-inputs.json",dict(kind="POSTHOC_CONSUMED_DEVELOPMENT_NO_NEW_OBSERVATIONS",old_hashes=old_hashes))
        for path in (Path(__file__),Path(guide.__file__),Path(guide.__file__).with_name("test_witness_guided_path.py"),
            Path(openloop.__file__),Path(openloop.__file__).with_name("test_two_step_openloop.py"),Path(two.__file__),
            Path(model.__file__),Path(model.__file__).with_name("test_path_constraint_inference.py"),
            Path(oldwitness.__file__),Path(av.__file__),Path(oldpos.__file__),
            DOCS/"OBSERVATION_DEEPENING_PROTOCOL_20260922.md",DOCS/"PATH_CONSTRAINT_BRIEF_20260922.md",DOCS/"TWO_STEP_OPENLOOP_BRIEF_20260922.md"):
            shutil.copyfile(path,output/path.name)
        av.seal(output/"pre-readout-seal.json",{p.name:p for p in output.iterdir() if p.is_file()})
        initial=read(OLD/"initial-observations.json")
        mapping={r["id"]:r["queries"]["initial"] for r in read(OLD/"witness-query-mapping.json")}
        initial_witness={qid:read(OLD/f"witness-{qid}.json") for qid in set(mapping.values())}
        assert all(len(r["observations"])==1 and r["observations"][0]["camera"]==[0.,0.] for r in initial_witness.values())
        public=read(OLD/"tree-public-forecasts.json")
        forecasts={tuple(r["camera"]):tuple(map(tuple,r["bins"])) for r in public["poses"]}
        labels=tuple(public["labels"])
        choices=[]
        for row in initial:
            answer=initial_witness[mapping[row["id"]]]
            assert answer["observations"][0]["bins"]==row["bins"]
            choices.append(dict(id=row["id"],initial_witness_query=mapping[row["id"]],
                guide=guide.choose(row["bins"],answer["result"]),openloop=openloop.choose(row["bins"],forecasts,labels)))
        av.write_json(output/"sealed-choices.json",choices)
        av.seal(output/"choice-seal.json",dict(choices=output/"sealed-choices.json",initial=OLD/"initial-observations.json",
            witness_mapping=OLD/"witness-query-mapping.json",prior=OLD/"tree-public-forecasts.json",readout=output/"pre-readout-seal.json"))
        stage("public_choices_sealed_before_actual_future_trace_parse")
        av.verify_seal(output/"choice-seal.json")
        choices=read(output/"sealed-choices.json")
        observations={r["id"]:{tuple(v["camera"]):v for v in r["views"]} for r in read(OLD/"all-pose-observations.json")}
        oldfirst={r["id"]:r for r in read(OLD/"first-choices.json")}
        oldsecond={r["id"]:r for r in read(OLD/"second-choices.json")}
        queries={}
        selected=[]
        finite_predictions=[]
        for row in choices:
            ident=row["id"]
            mid=two.STEPS[row["openloop"]["first_action"]]
            final=two.add(mid,two.STEPS[row["openloop"]["second_action"]])
            feedback_mid=two.STEPS[oldfirst[ident]["lookahead"]["action"]]
            paths=dict(endpoint=(av.ORIGIN,av.ACTIONS[0]),fixed_path=guide.poses(0),guided_path=guide.poses(row["guide"]["action"]),
                feedback=(av.ORIGIN,feedback_mid,tuple(oldsecond[ident]["lookahead_final_pose"])),openloop=(av.ORIGIN,mid,final))
            query_map={}
            for mode,path in paths.items():
                obs=[dict(camera=list(p),bins=observations[ident][p]["bins"]) for p in path]
                key=av.canonical(obs).decode()
                if key not in queries:
                    queries[key]=dict(query_id=f"query_{len(queries):03d}",observations=obs)
                query_map[mode]=queries[key]["query_id"]
            support=row["openloop"]["candidates"]
            for p in (mid,final):
                support=two.update(support,observations[ident][p]["bins"],p,forecasts)
            finite_predictions.append(dict(id=ident,decision=two.decision(support,labels),remaining=list(support),path=[av.ORIGIN,mid,final]))
            selected.append(dict(id=ident,queries=query_map,paths=paths))
        av.write_json(output/"selected-traces.json",selected)
        av.write_json(output/"finite-openloop-predictions.json",finite_predictions)
        av.write_json(output/"continuous-public-queries.json",list(queries.values()))
        av.seal(output/"solver-input-seal.json",dict(queries=output/"continuous-public-queries.json",selected=output/"selected-traces.json",
            finite=output/"finite-openloop-predictions.json",choices=output/"choice-seal.json",observations=OLD/"all-pose-observations.json"))
        stage("all_actual_trace_inputs_and_finite_predictions_sealed_before_truth")
        answers=[]
        for i,q in enumerate(queries.values()):
            result=model.infer(q["observations"])
            assert result["decision"] in MAP
            answer=dict(**q,result=result)
            answers.append(answer)
            av.write_json(output/f"{q['query_id']}.json",answer)
            if i%20==0 or i==len(queries)-1:
                print(json.dumps(dict(solved=i+1,total=len(queries),seconds=time.perf_counter()-start)),flush=True)
        av.write_json(output/"continuous-predictions.json",answers)
        av.seal(output/"prediction-seal.json",{p.name:p for p in output.iterdir() if p.is_file()})
        stage("all_continuous_and_finite_predictions_sealed_before_truth_join")
        av.verify_seal(output/"prediction-seal.json")
        truth={r["id"]:av.intersects_query(oldpos.source_scene(r)) for r in read(OLD/"source.json")}
        by_query={a["query_id"]:a for a in answers}
        old_predictions={r["id"]:r["policies"] for r in read(OLD/"policy-predictions.json")}
        by_finite={r["id"]:r for r in finite_predictions}
        rows=[]
        for row in selected:
            ident=row["id"]
            values={m:MAP[by_query[q]["result"]["decision"]] for m,q in row["queries"].items()}
            rows.append(dict(id=ident,truth=truth[ident],**values,finite_openloop=by_finite[ident]["decision"],
                finite_feedback=old_predictions[ident]["lookahead_two_step"]["decision"],
                finite_fixed=old_predictions[ident]["fixed_two_step"]["decision"]))
        metrics={m:counts(rows,m) for m in (*MODES,"finite_openloop","finite_feedback","finite_fixed")}
        contrasts={f"{a}_versus_{b}":contrast(rows,a,b) for a,b in (("fixed_path","endpoint"),("guided_path","fixed_path"),
            ("finite_openloop","finite_feedback"),("finite_openloop","finite_fixed"),("feedback","finite_feedback"),("openloop","finite_openloop"))}
        reasons={mode:dict(Counter(by_query[r["queries"][mode]]["result"]["reason"] for r in selected)) for mode in MODES}
        witness_counts={mode:dict(Counter(str(sum(v is not None for v in by_query[r["queries"][mode]]["result"]["witnesses"].values())) for r in selected)) for mode in MODES}
        information={}
        for mode in MODES:
            buckets=defaultdict(list)
            for r in selected:
                # Public camera coordinates are part of an adaptive trace signature.
                signature=av.canonical(by_query[r["queries"][mode]]["observations"])
                buckets[signature].append(r["id"])
            pure=[i for group in buckets.values() if len({truth[i] for i in group})==1 for i in group]
            information[mode]=dict(pure_count=len(pure),pure_positive=sum(truth[i] for i in pure),pure_ids=sorted(pure))
        checks=0
        for answer in answers:
            for label,w in answer["result"]["witnesses"].items():
                if w is None:
                    continue
                scene=oldpos.source_scene(dict(scene=w))
                assert av.intersects_query(scene)==(label=="IN")
                assert all(av.observe(scene,tuple(o["camera"]))==tuple(o["bins"]) for o in answer["observations"])
                checks+=1
        summary=dict(kind="CONSUMED_SAVED_TRACE_DEVELOPMENT",unique_queries=len(queries),metrics=metrics,contrasts=contrasts,
            reason_counts=reasons,witness_counts=witness_counts,finite_cohort_information=information,validated_witnesses=checks,
            guide_actions=dict(Counter(str(r["guide"]["action"]) for r in choices)),
            guide_reasons=dict(Counter(r["guide"]["reason"] for r in choices)),
            costs={m:dict(views=13 if m.endswith("path") else 2 if m=="endpoint" else 3,motion_m=.12) for m in MODES})
        av.write_json(output/"evaluation.json",rows)
        av.write_json(output/"summary.json",summary)
        stage("evaluation_complete")
        assert all(av.file_hash(Path(p))==h for p,h in old_hashes.items())
        av.write_json(output/"stage-order.json",stages)
        av.write_json(output/"local-disposition.json",dict(terminal_id="ba-observation-deepening-20260922",role="COMPONENT_OR_CHALLENGER",mode="COMPONENT",
            scope="Consumed path constraints, witness-guided replay and openloop comparison; numerical model-only outputs",
            global_registration="PENDING_SUPPORTED_COMMAND",persistent_resources=[]))
        av.seal(output/"completion-seal.json",{p.name:p for p in output.iterdir() if p.is_file()})
        print(json.dumps(dict(status="COMPLETED",queries=len(queries),metrics=metrics,seconds=stages[-1]["seconds"])),flush=True)
    except BaseException as exc:
        av.write_json(output/"failure.json",dict(type=type(exc).__name__,message=str(exc),stages=stages))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    canonical=(ROOT/"artifacts.local/work/ba-observation-deepening-20260922").resolve()
    if canonical not in args.output.resolve().parents:
        parser.error("Output must be new child of canonical deepening artifact root")
    run(args.output)
