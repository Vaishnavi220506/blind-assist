"""Independent saved-record audit; no fit, image/trunk inference, or cutoff search."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "artifacts.local/work/ba-last-layer-20260921"
OLD = ROOT / "artifacts.local/work/ba-spatial-bce-20260920"
TRANSFER = ROOT / "artifacts.local/work/ba-spatial-complement-transfer-20260921"
ROLES = ("fit", "selection", "evaluation")
ARMS = ("A", "C", "uniform", "balanced")
HIGH = 7.6612162590026855


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a, b, tol=1e-10):
    if a is None or b is None:
        assert a is b, (a, b)
    else:
        assert abs(a-b) <= tol, (a, b)


def by_clip(rows):
    clips = defaultdict(list)
    for row in rows:
        clips[row["clip_id"]].append(row)
    return [sorted(c, key=lambda r:r["frame_in_clip"]) for c in clips.values()]


def count_runs(flags):
    return sum(flag and (i == 0 or not flags[i-1]) for i, flag in enumerate(flags))


def count_metrics(rows, name):
    t = [r["truth"] for r in rows]
    a = [r["flags"][name] for r in rows]
    u = [r["unknown"] for r in rows]
    pos = sum(t)
    neg = len(rows)-pos
    tp = sum(y and p for y,p in zip(t,a))
    fp = sum(not y and p for y,p in zip(t,a))
    fn = pos-tp
    tn = sum(not y and not p and not q for y,p,q in zip(t,a,u))
    frames = dict(TP=tp, FP=fp, FN=fn, TN=tn,
        positive_frames=pos, negative_frames=neg, current_unknown=sum(u),
        abstained_positive=sum(y and not p and q for y,p,q in zip(t,a,u)),
        abstained_negative=sum(not y and not p and q for y,p,q in zip(t,a,u)),
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/pos if pos else None, FPR=fp/neg if neg else None)
    segments = 0
    events = []
    for clip in by_clip(rows):
        alerts = [r["flags"][name] for r in clip]
        segments += count_runs([not r["truth"] and p for r,p in zip(clip,alerts)])
        positive = [i for i,r in enumerate(clip) if r["truth"]]
        if not positive:
            continue
        start, end = positive[0],positive[-1]+1
        assert positive == list(range(start,end))
        event = alerts[start:end]
        found = [i for i,v in enumerate(event) if v]
        first = found[0] if found else None
        last = found[-1] if found else None
        internal_silence = sum(not v for v in event[first:last+1]) if found else 0
        post = alerts[end:]
        first_silent = next((i for i,v in enumerate(post) if not v),None)
        leading = first_silent if first_silent is not None else len(post)
        events.append(dict(clip_id=clip[0]["clip_id"], detected=bool(found),
            first_in_event_alert_delay_s=.2*first if first is not None else None,
            positive_alert_frames=sum(event), positive_frames=len(event),
            positive_coverage=sum(event)/len(event),
            initial_silent_frames=first if first is not None else len(event),
            terminal_silent_frames=len(event)-last-1 if last is not None else 0,
            internal_silent_frames=internal_silence,
            preentry_false_alert_frames=sum(alerts[:start]),
            postexit_false_alert_frames=sum(post),
            first_silent_relative_to_exit_s=.2*first_silent if first_silent is not None else None,
            postexit_carryover_alert_frames=leading if event[-1] else 0))
    return dict(frames=frames,false_alert_segment_count=segments,
        event_count=len(events),detected_events=sum(e["detected"] for e in events),events=events)


def compare_metric(actual, saved):
    for key,value in actual["frames"].items():
        close(value,saved["frames"][key])
    for key in ("false_alert_segment_count","event_count","detected_events"):
        assert actual[key] == saved[key], key
    se={e["clip_id"]:e for e in saved["events"]}
    for event in actual["events"]:
        for key,value in event.items():
            if key=="clip_id": continue
            close(value,se[event["clip_id"]][key])


def xauc_recount(rows):
    groups=sorted({r["base_group_id"] for r in rows})
    paired=defaultdict(dict)
    for r in rows:
        paired[(r["base_group_id"],r["frame_in_clip"])][r["layout_relation"]]=r
    pairs=[p for p in paired.values() if p["BOUNDARY"]["truth"] and p["INSIDE"]["truth"] and not p["OUTSIDE"]["truth"]]
    result={}
    for arm in ("original","uniform","balanced"):
        matrix=[];counts=[]
        for g in groups:
            positives=[p["BOUNDARY"]["scores"][arm] for p in pairs if p["BOUNDARY"]["base_group_id"]==g]
            line=[];nline=[]
            for h in groups:
                negatives=[p["OUTSIDE"]["scores"][arm] for p in pairs if p["OUTSIDE"]["base_group_id"]==h]
                comparisons=[float(p>n)+.5*float(p==n) for p in positives for n in negatives]
                line.append(sum(comparisons)/len(comparisons) if comparisons else None)
                nline.append(len(comparisons))
            matrix.append(line);counts.append(nline)
        off=[v for i,line in enumerate(matrix) for j,v in enumerate(line) if i!=j and v is not None]
        result[arm]=dict(groups=groups,matrix=matrix,pair_counts=counts,
            off_diagonal_equal_layout_mean=sum(off)/len(off),off_diagonal_min=min(off),
            primary_matched_pairs=len(pairs),
            primary_matched_ordered=sum(p["BOUNDARY"]["scores"][arm]>p["OUTSIDE"]["scores"][arm] for p in pairs))
    return result


def main():
    assert not (OUT/"independent-audit.json").exists(), "Do not overwrite completed audit"
    protocol=read(OUT/"protocol.json")
    checked=0
    for namespace in ("sources","code"):
        for path,digest in protocol[namespace].items():
            assert sha(ROOT/path)==digest,path
            checked+=1
    assert (OUT/"protocol-before-run.md").read_bytes()==(HERE/"LAST_LAYER_PROTOCOL_20260921.md").read_bytes()
    seal_names=("role","hidden","fit","prediction","evaluation")
    for name in seal_names:
        seal=read(OUT/(name+"-seal.json"))
        assert seal["protocol_sha256"]==sha(OUT/"protocol.json")
        for path,digest in seal["hashes"].items():
            assert sha(OUT/path)==digest,path
            checked+=1
    forbidden_test_files=("test-start.json","test-logits.npy","test-prediction-seal.json","test-metrics.json")
    assert not any((OLD/name).exists() for name in forbidden_test_files)
    assert protocol["scope"]=="CONSUMED_DEVELOPMENT"
    assert protocol["challenger"]=="balanced" and protocol["control"]=="uniform"
    source_data={}
    for tag,source,label_file in (("dev",OLD,"dev-labels.json"),("transfer",TRANSFER,"transfer-labels.json")):
        metas=read(source/"evaluator/metadata.json")
        labels={r["index"]:r["truth"] for r in read(source/"evaluator"/label_file)}
        ids=read(source/"identities.json")
        cases=read(source/"spec.json")["cases"]
        source_data[tag]=dict(meta={r["index"]:r for r in metas},
            labels=labels,baseline=read(source/"baseline.json"),ids=ids,
            native={r["id"]:r for r in read(source/"source-admission.json")["frames"]},cases=cases)
    all_rows={};groups={};objective_checks={}
    fit_hidden=np.load(OUT/"fit-hidden.npy").astype(np.float64)
    reference_mean=fit_hidden.mean(axis=0)
    reference_scale=np.maximum(fit_hidden.std(axis=0),1e-6)
    fit_metadata=read(OUT/"fit-rows.json")
    fit_y=np.array([r["truth"] for r in read(OUT/"labels/fit.json")],dtype=float)
    assert fit_hidden.shape==(576,32)
    cell_counts=Counter((r["base_group_id"],bool(y)) for r,y in zip(fit_metadata,fit_y))
    assert len(cell_counts)==16
    standardized=(fit_hidden-reference_mean)/reference_scale
    for arm in ("uniform","balanced"):
        with np.load(OUT/(arm+"-readout.npz")) as readout:
            w,b=readout["weight"],float(readout["bias"])
            assert w.shape==(32,) and readout["bias"].shape==()
            np.testing.assert_array_equal(reference_mean,readout["mean"])
            np.testing.assert_array_equal(reference_scale,readout["scale"])
            weights=np.array([1/(16*cell_counts[(r["base_group_id"],bool(y))]) for r,y in zip(fit_metadata,fit_y)]) if arm=="balanced" else np.full(576,1/576)
            close(float(weights.sum()),1.)
            logits=standardized@w+b
            errors=np.exp(-np.logaddexp(0,-logits))-fit_y
            gradient=np.r_[standardized.T@(weights*errors)+.001*w,np.sum(weights*errors)]
            objective=float(weights@(np.logaddexp(0,logits)-fit_y*logits)+.001/2*(w@w))
            receipt=read(OUT/(arm+"-fit.json"))
            close(objective,receipt["objective"])
            close(float(abs(gradient).max()),receipt["gradient_max"],1e-10)
            assert receipt["parameters"]==33
            assert receipt["correct_at_zero"]==int(np.sum((logits>=0)==fit_y))
            close(weights.min(),receipt["weight_min"])
            close(weights.max(),receipt["weight_max"])
            objective_checks[arm]=dict(parameters=33,objective=objective,
                independent_gradient_max=float(abs(gradient).max()),cell_counts={g+":"+str(y):n for (g,y),n in cell_counts.items()},
                preprocessing="FIT-only population mean/std exactly reproduced")
    selection=read(OUT/"selection.json")
    original_raw={"dev":np.load(OLD/"dev-logits.npy"),"transfer":np.load(TRANSFER/"logits.npy")}
    for role in ROLES:
        metadata=read(OUT/(role+"-rows.json"))
        labels=read(OUT/"labels"/(role+".json"))
        predictions=read(OUT/(role+"-predictions.json"))
        saved_rows=read(OUT/(role+"-frame-results.json"))
        hidden=np.load(OUT/(role+"-hidden.npy")).astype(np.float64)
        original=np.load(OUT/(role+"-original.npy"))
        assert len(metadata)==len(labels)==len(predictions)==len(saved_rows)==576
        assert [r["id"] for r in metadata]==[r["id"] for r in labels]==[r["id"] for r in predictions]==[r["id"] for r in saved_rows]
        assert len(set(r["id"] for r in metadata))==576
        groups[role]=set(r["base_group_id"] for r in metadata)
        assert len(groups[role])==8
        assert set(Counter(r["type_id"] for r in metadata).values())=={144}
        scores={}
        for arm in ("uniform","balanced"):
            with np.load(OUT/(arm+"-readout.npz")) as readout:
                scores[arm]=((hidden-readout["mean"])/readout["scale"])@readout["weight"]+float(readout["bias"])
            np.testing.assert_allclose(scores[arm],[p["scores"][arm] for p in predictions],rtol=0,atol=1e-12)
        for i,(r,label,p,saved) in enumerate(zip(metadata,labels,predictions,saved_rows)):
            source=source_data[r["source"]]
            index=r["index"]
            raw=source["meta"][index]
            assert r["source_id"]==raw["id"]==source["ids"][index]["id"]
            assert r["id"]==r["source"]+":"+r["source_id"]
            assert r["clip_id"]==raw["clip_id"]
            assert label["truth"]==source["labels"][index]==saved["truth"]
            assert r["a"]==source["baseline"][index]["current"]
            assert r["unknown"]==source["baseline"][index]["unknown"]
            if role=="fit":
                assert r["source"]=="dev" and raw["split"]=="dev"
                close(original[i],original_raw["dev"][i])
            else:
                assert r["source"]=="transfer"
                suffix=r["base_group_id"].rsplit("_",1)[-1]
                assert suffix in (("g00","g01") if role=="selection" else ("g02","g03"))
                close(original[i],original_raw["transfer"][index])
            close(p["scores"]["original"],original[i])
            expected=dict(A=bool(r["a"]),C=bool(r["a"] or original[i]>=HIGH))
            expected.update({arm:bool(r["a"] or scores[arm][i]>=selection[arm]["threshold"]) for arm in scores})
            for arm,value in expected.items():
                assert p["flags"][arm+"_current"]==value
                assert p["current_unknown"][arm+"_current"]==r["unknown"]==p["current_unknown"][arm+"_hold"]
                assert saved["flags"][arm+"_current"]==value
            native=source["native"][r["source_id"]]["returned_target_corridor_samples"]
            assert saved["native_target_corridor_samples"]==native
            assert saved["scores"]==p["scores"]
        rows=saved_rows
        for clip in by_clip(rows):
            assert len(clip)==24 and [r["frame_in_clip"] for r in clip]==list(range(24))
            for i,r in enumerate(clip):
                close(r["time_s"],.2*i)
                for arm in ARMS:
                    expected=r["flags"][arm+"_current"] or (i>0 and clip[i-1]["flags"][arm+"_current"])
                    assert r["flags"][arm+"_hold"]==bool(expected)
        all_rows[role]=rows
    assert all(not groups[a]&groups[b] for i,a in enumerate(ROLES) for b in ROLES[i+1:])
    selection_rows=all_rows["selection"]
    forbidden=set()
    for clip in by_clip(selection_rows):
        for i,row in enumerate(clip):
            if not row["truth"] and not row["flags"]["A_current"]:
                forbidden.add(row["id"])
            if i>0 and not row["truth"] and not row["flags"]["A_hold"]:
                forbidden.add(clip[i-1]["id"])
    threshold_checks={}
    for arm in ("uniform","balanced"):
        maximum=max(r["scores"][arm] for r in selection_rows if r["id"] in forbidden)
        assert set(selection[arm]["forbidden_ids"])==forbidden
        assert selection[arm]["threshold"]==float(np.nextafter(maximum,np.inf))
        assert not any(r["flags"][arm+"_"+suffix] and not r["flags"]["A_"+suffix] and not r["truth"] for r in selection_rows for suffix in ("current","hold"))
        blockers=[]
        for clip in by_clip(selection_rows):
            for i,r in enumerate(clip):
                if r["id"] in forbidden and r["scores"][arm]==maximum:
                    next_row=clip[i+1] if i+1<len(clip) else None
                    blockers.append(dict(id=r["id"],group=r["base_group_id"],relation=r["layout_relation"],
                        time_s=r["time_s"],truth=r["truth"],a_current=r["flags"]["A_current"],
                        forbidden_due_to_current_negative=not r["truth"] and not r["flags"]["A_current"],
                        forbidden_due_to_next_held_negative=bool(next_row and not next_row["truth"] and not next_row["flags"]["A_hold"]),
                        next_id=next_row["id"] if next_row else None))
        threshold_checks[arm]=dict(threshold=selection[arm]["threshold"],forbidden_count=len(forbidden),forbidden_max=maximum,max_blockers=blockers)
    detailed={};xa={}
    for role,rows in all_rows.items():
        metrics=read(OUT/(role+"-metrics.json"))
        group_metrics=read(OUT/(role+"-group-metrics.json"))
        checks={};native_changes={};depth_fp={};timing={};worst={}
        for partition in ("Core","Boundary"):
            subset=[r for r in rows if (r["layout_relation"]=="BOUNDARY")==(partition=="Boundary")]
            checks[partition]={};timing[partition]={}
            for arm in ARMS:
                for suffix in ("current","hold"):
                    name=arm+"_"+suffix
                    m=count_metrics(subset,name)
                    compare_metric(m,metrics[partition]["arms"][name])
                    checks[partition][name]={k:m["frames"][k] for k in ("TP","FP","FN","TN","current_unknown","precision","recall","FPR")}
                    timing[partition][name]=m["events"]
        for group in sorted(groups[role]):
            subset=[r for r in rows if r["base_group_id"]==group]
            for partition in ("Core","Boundary"):
                part=[r for r in subset if (r["layout_relation"]=="BOUNDARY")==(partition=="Boundary")]
                for arm in ARMS:
                    for suffix in ("current","hold"):
                        name=arm+"_"+suffix
                        compare_metric(count_metrics(part,name),group_metrics[group][partition]["arms"][name])
        for arm in ARMS:
            for suffix in ("current","hold"):
                name=arm+"_"+suffix
                for partition in ("Core","Boundary"):
                    entries=[(g,group_metrics[g][partition]["arms"][name]["frames"]) for g in sorted(groups[role])]
                    recalls=[(g,f["recall"]) for g,f in entries if f["recall"] is not None]
                    fprs=[(g,f["FPR"]) for g,f in entries if f["FPR"] is not None]
                    minr=min(v for g,v in recalls)
                    maxf=max(v for g,v in fprs)
                    worst[partition+"_"+name]=dict(min_recall=minr,min_recall_groups=[g for g,v in recalls if v==minr],
                        max_FPR=maxf,max_FPR_groups=[g for g,v in fprs if v==maxf])
        for arm in ("uniform","balanced"):
            native_changes[arm]={}
            for suffix in ("current","hold"):
                native_changes[arm][suffix]={}
                for label,condition in (
                    ("new_A_true_frames",lambda r:r["truth"] and r["flags"][arm+"_"+suffix] and not r["flags"]["A_"+suffix]),
                    ("lost_C_true_frames",lambda r:r["truth"] and r["flags"]["C_"+suffix] and not r["flags"][arm+"_"+suffix])):
                    chosen=[r for r in rows if condition(r)]
                    native_changes[arm][suffix][label]=dict(count=len(chosen),native_backed=sum(r["native_target_corridor_samples"]>0 for r in chosen),
                        boundary_count=sum(r["layout_relation"]=="BOUNDARY" for r in chosen),
                        boundary_native_backed=sum(r["layout_relation"]=="BOUNDARY" and r["native_target_corridor_samples"]>0 for r in chosen),
                        ids=[r["id"] for r in chosen])
        for arm in ARMS:
            for suffix in ("current","hold"):
                name=arm+"_"+suffix
                values=Counter()
                for r in rows:
                    if r["truth"] or not r["flags"][name]:
                        continue
                    case=source_data[r["source"]]["cases"][r["index"]]
                    camera=case["camera"]
                    assert all(camera[k]==0 for k in ("pitch","yaw","roll"))
                    target=next(o for o in case["objects"] if o["name"]==case["target_name"])
                    near=target["center_m"][0]-target["size_m"][0]/2-camera["x"]
                    far=target["center_m"][0]+target["size_m"][0]/2-camera["x"]
                    category="axial_depth_outside_contract" if near>3. or far<.3 else "within_depth_outside_corridor"
                    values[category]+=1
                    values[r["layout_relation"]+":"+category]+=1
                depth_fp[name]=dict(values)
        xr=xauc_recount(rows);xs=read(OUT/(role+"-xauc.json"))
        for arm in xr:
            assert xr[arm]["groups"]==xs[arm]["groups"]
            assert xr[arm]["pair_counts"]==xs[arm]["pair_counts"]
            np.testing.assert_allclose(xr[arm]["matrix"],xs[arm]["matrix"],atol=1e-14,rtol=0)
            for k in ("off_diagonal_equal_layout_mean","off_diagonal_min","primary_matched_pairs","primary_matched_ordered"):
                close(xr[arm][k],xs[arm][k])
        xa[role]=xr
        detailed[role]=dict(counts=checks,all_actual_worst_groups=worst,native_changes=native_changes,
            false_positive_geometry=depth_fp,independent_event_timing=timing,
            exact_A_flags={arm:all(r["flags"][arm+"_"+s]==r["flags"]["A_"+s] for r in rows for s in ("current","hold")) for arm in ("uniform","balanced")})
    transfer_x=xauc_recount(all_rows["selection"]+all_rows["evaluation"])
    saved_transfer=read(OUT/"transfer-xauc.json")
    for arm in transfer_x:
        np.testing.assert_allclose(transfer_x[arm]["matrix"],saved_transfer[arm]["matrix"],atol=1e-14,rtol=0)
        close(transfer_x[arm]["off_diagonal_equal_layout_mean"],saved_transfer[arm]["off_diagonal_equal_layout_mean"])
    result=read(OUT/"result.json")
    gate_checks={}
    for role,rows in all_rows.items():
        gate_checks[role]={}
        for arm in ("uniform","balanced"):
            boundary=[r for r in rows if r["layout_relation"]=="BOUNDARY"]
            rescue=[r for r in boundary if r["truth"] and r["flags"]["C_current"] and not r["flags"]["A_current"]]
            retained=sum(r["flags"][arm+"_current"] for r in rescue)
            noextra=not any(not r["truth"] and r["flags"][arm+"_"+s] and not r["flags"]["A_"+s] for r in rows for s in ("current","hold"))
            a_retained=all(not r["flags"]["A_"+s] or r["flags"][arm+"_"+s] for r in rows for s in ("current","hold"))
            gains=[r for r in boundary if r["truth"] and r["flags"][arm+"_current"] and not r["flags"]["A_current"]]
            gaining_groups=len({r["base_group_id"] for r in gains})
            gain=len(gains)/sum(r["truth"] for r in boundary)
            c_events={r["clip_id"] for r in boundary if r["truth"] and r["flags"]["C_current"]}
            new_events={r["clip_id"] for r in boundary if r["truth"] and r["flags"][arm+"_current"]}
            event_retention=c_events<=new_events
            rate=retained/len(rescue) if rescue else None
            useful=a_retained and noextra and gain>=.10 and gaining_groups>=4 and rate is not None and rate>=.90 and event_retention
            recomputed=dict(all_A_flags_retained=a_retained,no_added_FP_current_and_hold=noextra,C_boundary_rescues=len(rescue),
                C_boundary_rescues_retained=retained,C_rescue_retention=rate,C_boundary_detected_events_retained=event_retention,
                boundary_recall_gain=gain,gaining_groups=gaining_groups,useful_repair=bool(useful))
            for key,value in recomputed.items():
                close(value,result["gates"][role][arm][key])
            gate_checks[role][arm]=recomputed
    for arm in ("uniform","balanced"):
        assert result[arm+"_useful_repair"]==all(gate_checks[role][arm]["useful_repair"] for role in ("selection","evaluation"))
    assert not any((OLD/name).exists() for name in forbidden_test_files)
    audit=dict(status="PASS",scope="Independent saved-record audit, consumed Development",audit_script_sha256=sha(__file__),
        verified_hash_entries=checked,verified_seals=list(seal_names),original_test_closed=True,
        no_fit_or_trunk_inference=True,no_new_operating_point=True,
        role_groups={k:sorted(v) for k,v in groups.items()},objective_and_weight_checks=objective_checks,
        cutoff_recount=threshold_checks,gates=gate_checks,roles=detailed,xauc=xa,transfer_xauc=transfer_x,
        limits=["Frozen-feature extraction itself was not rerun; original model/cache integrity and recorded extraction receipts remain source evidence.",
                "Evaluation labels were materialized in prepare as disclosed; this is not fresh confirmation.",
                "Geometry FP partition uses complete target axial extent and camera yaw/pitch/roll zero; no new metric labels enter inference."])
    (OUT/"independent-audit.json").write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="PASS",verified_hash_entries=checked,evaluation_counts=detailed["evaluation"]["counts"],
        evaluation_gates=gate_checks["evaluation"],evaluation_geometry_fp=detailed["evaluation"]["false_positive_geometry"],
        evaluation_xauc={k:{m:v[m] for m in ("off_diagonal_equal_layout_mean","off_diagonal_min","primary_matched_ordered","primary_matched_pairs")} for k,v in xa["evaluation"].items()}),indent=2))


if __name__=="__main__":
    main()
