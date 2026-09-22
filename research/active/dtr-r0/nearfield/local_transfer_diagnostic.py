"""Read-only consumed LOCAL error diagnosis; no model load, fit or cutoff search.

Use only through the governed research-ue RunSpec. Every saved evaluation frame
is retained. Outcome-defined groups are descriptive and cannot establish causes
or select a new candidate. The exported gallery is a scientific input preview.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from tools.research_backend import BackendCandidate, DeviceObservation, select_backend

RAW_SIZE = 910
CENTRE = [1, 4]
VISUAL = ('red_mean', 'green_mean', 'blue_mean', 'red_std', 'green_std',
          'blue_std', 'grad_x_mean', 'grad_y_mean', 'gray_q10', 'gray_q90')
BANDS = ('definite', 'possible_only', 'outside')
GEOMETRY = ('range_low_mean_m', 'range_high_mean_m', 'range_mean_m',
            'range_std_m', 'pixel_fraction')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def describe_feature(feature):
    output = {}
    for band, start in zip(BANDS, (910, 925, 940), strict=True):
        for j, key in enumerate(VISUAL):
            output[band+'.'+key] = float(feature[start+j])
        for j, key in enumerate(GEOMETRY):
            output[band+'.'+key] = float(feature[start+10+j] * (8 if j < 4 else 1))
    for j, key in enumerate(VISUAL[:3]):
        output['definite_minus_outside.'+key] = float(feature[955+j])
        output['possible_minus_outside.'+key] = float(feature[958+j])
    for j, key in enumerate(VISUAL):
        output['whole.'+key] = float(feature[896+j])
    return output


def distribution(records):
    if not records:
        return dict(n=0, features={})
    keys = tuple(records[0]['features'])
    return dict(n=len(records), features={k:dict(zip(('min', 'q05', 'median', 'q95', 'max'),
        np.quantile([r['features'][k] for r in records], [0, .05, .5, .95, 1]).tolist(), strict=True)) for k in keys},
        local_score=dict(zip(('min', 'q05', 'median', 'q95', 'max'),
            np.quantile([r['local_score'] for r in records], [0, .05, .5, .95, 1]).tolist(), strict=True)),
        raw_score=dict(zip(('min', 'q05', 'median', 'q95', 'max'),
            np.quantile([r['raw_score'] for r in records], [0, .05, .5, .95, 1]).tolist(), strict=True)),
        by_type=dict(Counter(r['type_id'] for r in records)),
        by_relation=dict(Counter(r['layout_relation'] for r in records)),
        by_query=dict(Counter(str(r['local_winning_query']) for r in records)),
        layouts=len({r['base_group_id'] for r in records}))


def gallery(output, pairs, rgb, index):
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 14)
    w, h = 640, 426
    canvas = Image.new('RGB', (w*3, h*len(pairs)), 'white')
    draw = ImageDraw.Draw(canvas)
    for ri, pair in enumerate(pairs):
        for ci, row in enumerate(pair['same_layout_and_sample']):
            image = Image.fromarray(rgb[index[row['id']]].transpose(1, 2, 0)).resize((640, 360))
            left, top = ci*w, ri*h
            canvas.paste(image, (left, top))
            text = (f"{row['type_id']} {row['base_group_id']} f{row['frame_in_clip']:02d}\n"
                f"{row['layout_relation']} truth={row['truth']} A={row['A_current']} "
                f"LOCAL={row['local_score']:.5f} RAW={row['raw_score']:.5f}\n"
                f"q={row['local_winning_query']} possible_fraction="
                f"{row['features']['possible_only.pixel_fraction']:.4f}")
            draw.multiline_text((left+5, top+364), text, fill='black', font=font, spacing=1)
    canvas.save(output)


def run(args):
    out = args.result.parent.resolve()
    if not out.is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Output outside canonical artifact root')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new empty diagnostic output directory')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    run_files = ('frame-results.json', 'features.npz', 'selection.json', 'raw-predictions.npz',
                 'local-predictions.npz', 'metrics.json', 'freeze.json', 'feature-seal.json',
                 'model-seal.json', 'prediction-seal.json', 'result.json')
    inputs = {str(args.run/name):sha(args.run/name) for name in run_files}
    inputs.update({str(p):sha(p) for p in (args.identities, args.rgb, args.tof, args.materialization)})
    write(out/'input-hashes.json', inputs)
    write(out/'source-hashes.json', {str(Path(__file__)):sha(Path(__file__)),
        str(ROOT/'tools/research_backend.py'):sha(ROOT/'tools/research_backend.py')})
    select_backend('scalar-scoring', cpu=BackendCandidate('numpy-saved-feature-diagnostic', 'cpu',
        lambda:len(inputs), lambda _:DeviceObservation('cpu', platform.processor() or 'host CPU',
        'NumPy '+np.__version__)), cpu_reason='TASK_NOT_GPU_SUITABLE', record_path=out/'backend.json',
        capabilities=dict(model_loaded=False, fit=False, threshold_search=False,
                          workload='Saved feature summaries, identity joins and hashes'))

    frames = read(args.run/'frame-results.json')
    identities = read(args.identities)
    selection = read(args.run/'selection.json')
    metric = read(args.run/'metrics.json')
    features = dict(np.load(args.run/'features.npz', allow_pickle=False))
    predictions = {a:dict(np.load(args.run/(a+'-predictions.npz'), allow_pickle=False)) for a in ('raw', 'local')}
    rgb = np.load(args.rgb, mmap_mode='r', allow_pickle=False)
    tof = np.load(args.tof, mmap_mode='r', allow_pickle=False)
    schema = dict(frame_count=len(frames), identity_count=len(identities),
        frame_fields=sorted(frames[0]), identity_fields=sorted(identities[0]),
        feature_arrays={k:dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in features.items()},
        prediction_arrays={a:{k:dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in p.items()} for a,p in predictions.items()},
        rgb=dict(shape=list(rgb.shape), dtype=str(rgb.dtype)), tof=dict(shape=list(tof.shape), dtype=str(tof.dtype)))
    write(out/'schema.json', schema)
    assert len(frames)==576 and len(identities)==1728
    assert set(features)=={'raw', 'local'} and all(v.shape==(1728, 6, 961) for v in features.values())
    assert rgb.shape==(1728, 3, 180, 320) and rgb.dtype==np.uint8 and tof.shape==(1728, 64, 6)
    assert np.array_equal(features['raw'][:,:,:RAW_SIZE], features['local'][:,:,:RAW_SIZE])
    assert np.count_nonzero(features['raw'][:,:,RAW_SIZE:])==0
    feature_seal = read(args.run/'feature-seal.json')
    pred_seal = read(args.run/'prediction-seal.json')
    source_freeze = read(args.run/'freeze.json')
    materialization = read(args.materialization)
    assert sha(args.run/'features.npz')==feature_seal['sha256']
    assert sha(args.run/'freeze.json')==feature_seal['freeze_sha256']==pred_seal['source_seal_sha256']
    assert sha(args.run/'selection.json')==pred_seal['selection_sha256']
    assert sha(args.run/'model-seal.json')==pred_seal['models_sha256']
    for name,p in [('rgb.npy',args.rgb),('tof.npy',args.tof),('identities.json',args.identities)]:
        assert sha(p)==source_freeze['input_hashes']['observations/'+name]==materialization['hashes']['observations/'+name]
    result = read(args.run/'result.json')
    assert sha(args.run/'metrics.json')==result['metrics_sha256']
    assert sha(args.run/'frame-results.json')==result['frame_results_sha256']
    assert sha(args.run/'prediction-seal.json')==result['prediction_seal_sha256']
    indices = np.array(feature_seal['indices']['evaluation'])
    assert len(indices)==576 and all(identities[i]['split']=='evaluation' for i in indices)
    cutoffs = {a:selection[a]['selection']['threshold'] for a in predictions}
    assert cutoffs==dict(raw=.776778260888226, local=.7959699076137833)
    for a,p in predictions.items():
        assert set(p)=={'indices', 'probability'} and p['probability'].shape==(576,6)
        assert np.array_equal(p['indices'], indices) and sha(args.run/(a+'-predictions.npz'))==pred_seal['predictions'][a]
    index = {m['id']:i for i,m in enumerate(identities)}
    records = []
    for ei, (row, ix) in enumerate(zip(frames, indices, strict=True)):
        meta = identities[ix]
        assert row['id']==meta['id'] and row['truth'] in (True,False)
        flags={a:row['predictions'][a]['alert'] for a in ('A_current','raw','local')}
        assert all(row['predictions'][a]['unknown']==meta['baseline']['unknown'] for a in row['predictions'])
        score = {a:float(predictions[a]['probability'][ei,CENTRE].max()) for a in predictions}
        win = {a:CENTRE[int(np.argmax(predictions[a]['probability'][ei,CENTRE]))] for a in predictions}
        assert all(flags[a]==bool(flags['A_current'] or score[a]>=cutoffs[a]) for a in predictions)
        ranges=tof[ix,:,0]*8; valid=(tof[ix,:,1]==1)&np.isfinite(ranges)&(ranges>=.1)&(ranges<8)
        record={k:row[k] for k in ('id','clip_id','frame_in_clip','time_s','base_group_id','type_id','layer','layout_relation','truth')}
        record.update(source_index=int(ix), **flags, unknown=meta['baseline']['unknown'], A_score=meta['baseline']['score'],
            local_score=score['local'], raw_score=score['raw'], local_margin=score['local']-cutoffs['local'],
            raw_margin=score['raw']-cutoffs['raw'], local_winning_query=win['local'], raw_winning_query=win['raw'],
            all_local_query_scores=predictions['local']['probability'][ei].tolist(),
            all_raw_query_scores=predictions['raw']['probability'][ei].tolist(),
            valid_zones=int(valid.sum()), observed_range_quantiles_m=np.quantile(ranges[valid],[0,.25,.5,.75,1]).tolist() if valid.any() else [],
            features=describe_feature(features['local'][ix,win['local']]),
            centre_query_features={str(q):describe_feature(features['local'][ix,q]) for q in CENTRE})
        records.append(record)
    groups = dict(all=records,
        positives=[r for r in records if r['truth']], negatives=[r for r in records if not r['truth']],
        A_missed_positive=[r for r in records if r['truth'] and not r['A_current']],
        local_rescued_TP=[r for r in records if r['truth'] and r['local'] and not r['A_current']],
        remaining_FN=[r for r in records if r['truth'] and not r['local']],
        A_silent_negative=[r for r in records if not r['truth'] and not r['A_current']],
        local_added_FP=[r for r in records if not r['truth'] and r['local'] and not r['A_current']],
        A_silent_negative_without_local_FP=[r for r in records if not r['truth'] and not r['A_current'] and not r['local']],
        local_vs_raw_gained_TP=[r for r in records if r['truth'] and r['local'] and not r['raw']],
        local_vs_raw_lost_TP=[r for r in records if r['truth'] and r['raw'] and not r['local']])
    expected=dict(all=576, positives=256, negatives=320, A_missed_positive=63,
        local_rescued_TP=30, remaining_FN=33, A_silent_negative=314, local_added_FP=2,
        A_silent_negative_without_local_FP=312, local_vs_raw_gained_TP=24, local_vs_raw_lost_TP=1)
    assert {k:len(v) for k,v in groups.items()}==expected
    assert [r['id'] for r in groups['local_rescued_TP']]==metric['comparisons']['local_vs_A_current']['gained_true_frames']
    assert [r['id'] for r in groups['local_added_FP']]==metric['comparisons']['local_vs_A_current']['added_false_frames']
    assert [r['id'] for r in groups['local_vs_raw_lost_TP']]==metric['comparisons']['local_vs_raw']['lost_true_frames']
    pairs=[]
    for critical in groups['local_added_FP']+groups['local_vs_raw_lost_TP']:
        siblings=[r for r in records if r['base_group_id']==critical['base_group_id'] and r['frame_in_clip']==critical['frame_in_clip']]
        assert len(siblings)==3 and {r['layout_relation'] for r in siblings}=={'INSIDE','BOUNDARY','OUTSIDE'}
        pairs.append(dict(anchor_id=critical['id'], same_layout_and_sample=sorted(siblings,key=lambda r:('INSIDE','BOUNDARY','OUTSIDE').index(r['layout_relation']))))
    distributions={k:distribution(v) for k,v in groups.items()}
    # Descriptive percentile ranks use the entire 576-frame evaluation denominator.
    for group in ('local_added_FP','local_vs_raw_lost_TP'):
        for r in groups[group]:
            r['feature_percentiles_all_evaluation']={k:sum(v['features'][k]<=value for v in records)/576 for k,value in r['features'].items()}
    write(out/'frame-diagnostic.json',records)
    write(out/'groups.json',{k:[r['id'] for r in v] for k,v in groups.items()})
    write(out/'feature-distributions.json',distributions)
    write(out/'critical-and-paired-rows.json',pairs)
    gallery(out/'paired-public-rgb.png',pairs,rgb,index)
    summary=dict(status='PASS', scope='READ_ONLY_CONSUMED_DEVELOPMENT_DIAGNOSTIC',
        denominators=expected, source_frames=1728, analyzed_frames=576, analyzed_queries=3456,
        evaluation_layouts=len({r['base_group_id'] for r in records}),
        local_cutoff=cutoffs['local'], raw_cutoff=cutoffs['raw'], unknown=sum(r['unknown'] for r in records),
        critical={k:[{a:r[a] for a in ('id','truth','local_score','raw_score','local_margin','raw_margin','local_winning_query','A_score','valid_zones','features')} for r in groups[k]]
                  for k in ('local_added_FP','local_vs_raw_lost_TP')},
        rescued=dict(n=30,by_type=dict(Counter(r['type_id'] for r in groups['local_rescued_TP'])),
            by_relation=dict(Counter(r['layout_relation'] for r in groups['local_rescued_TP'])),
            by_query=dict(Counter(r['local_winning_query'] for r in groups['local_rescued_TP'])),
            layouts=len({r['base_group_id'] for r in groups['local_rescued_TP']})),
        no_new_fit=True, no_model_loaded=True, no_threshold_search=True, no_new_predictions=True,
        true_return_ownership='NOT_OBSERVED', causal_attribution='NOT_ESTABLISHED',
        new_source_generalization='NOT_TESTED', elapsed_s=time.perf_counter()-started)
    assert all(sha(Path(p))==h for p,h in inputs.items())
    write(out/'summary.json',summary)
    write(out/'output-seal.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    write(args.result,dict(status='PASS',summary='summary.json',denominators=expected,unknown=487,
        output_seal_sha256=sha(out/'output-seal.json'),no_fit_or_new_prediction=True,
        evidence_boundary='Consumed controlled diagnostic, not fresh transfer or two-error separability evidence'))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--identities',type=Path,required=True)
    parser.add_argument('--rgb',type=Path,required=True)
    parser.add_argument('--tof',type=Path,required=True)
    parser.add_argument('--materialization',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        if args.result.parent.exists():
            write(args.result.parent/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
