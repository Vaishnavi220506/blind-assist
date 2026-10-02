"""Train/val-only native whole-query support, with unresolved depth semantics.

No sensor synthesis or model inference. Semantic references only diagnose the
visible support; they never define or filter the geometry query labels.
"""
import cnh_rgb_visible_query_gate as P
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
from PIL import Image
import cnh_rgb_sanpo_query as S
import cnh_rgb_range_anchor as A
import cnh_rgb_visible_query as V

ROOT=P.ROOT
DATA=ROOT/'artifacts.local/datasets/sanpo-synthetic-ba-nfo'
MANIFEST=ROOT/'artifacts.local/work/ba-nfo-20260919/prepared-manifest.json'
OUT=ROOT/'artifacts.local/work/cnh-rgb-sanpo-support-20261002'
RUN_ID='CNH_RGB_SANPO_SUPPORT_20261002'
HYPOTHESES=('optical_Z','radial')


def prepare(out):
    if (out/'PLAN.json').exists():
        raise FileExistsError('Preserve plan')
    prereg=[r for r in P.RUNS.read_text(encoding='utf8').splitlines() if RUN_ID in r]
    assert len(prereg)==1
    # Filter manifest before any source image/depth/mask/metadata access.
    inputs=[r for r in P.read(MANIFEST) if r['source']=='sanpo' and r['split'] in ('train','val')]
    assert Counter(r['split'] for r in inputs)=={'train':2000,'val':500}
    sessions=[]
    for scene in sorted({r['scene'] for r in inputs}):
        rows=sorted([r for r in inputs if r['scene']==scene],key=lambda r:r['id'])
        assert len(rows)==50 and len({r['split'] for r in rows})==1
        path=DATA/scene/'description.json'
        description=P.read(path)
        camera=description['session_camera_location'][0]
        assert camera in ('camera_head','camera_chest')
        assert all('/'+camera+'/' in r['depth'] for r in rows)
        sessions.append(dict(scene=scene,split=rows[0]['split'],camera=camera,
            intrinsics=description['session_camera_details'][0]['left_camera_params'],
            description_sha256=P.sha(path),rows=rows))
    plan=dict(run_id=RUN_ID,frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        preregistration_row=prereg[0],sessions=sessions,role='Consumed SANPO-Synthetic Development, train/val only; no test pixel access',
        hypotheses=HYPOTHESES,source_contract='Meters/float16 gzip confirmed by local cached docs; optical-Z versus radial, official pixel-center convention, body extrinsics unresolved. Two depth interpretations retained without outcome selection.',
        geometry='Public per-session pinhole intrinsics; integer pixel coordinates per existing SANPO reprojection implementation; camera x-right/y-down/Z-forward. Fixed45x45deg FOV; HEAD/BODY virtual camera queries whole0.6–2.1m, not wearer collision truth.',
        label='Same16-native-pixel D16 and95%possible-ray valid coverage as existing adapter; both depth interpretations independently. All contact/pass/clear/UNKNOWN retained. Pixel support differs physically across resolutions, not cross-source comparable incidence.',
        semantic='For frames with >=1 known contact query under either hypothesis, read existing panoptic mask. Red channel official semantic id; report histogram of contact points and class supports>=16 pixels. No mask filtering/relabeling/query modification. Missing masks retained.',
        interpretation='Support audit, no success/promotion threshold. Report shallow0-2/2-5 counts, sessions, camera/split, semantic composition, depth-hypothesis disagreement and FOV/UNKNOWN. Do not select positives for a subsequent unqualified benchmark; no inference before assessing support.',
        execution='CPU numpy/gzip, maximum2sessionworkers, BLAS1; no download/model/training/ToF synthesis',
        source_sha256={str(p.relative_to(ROOT)):P.sha(p) for p in (
            Path(__file__),Path(S.__file__),Path(A.__file__),Path(V.__file__),Path(P.__file__),MANIFEST,
            DATA/'official-taxonomy/labelmap.json')})
    P.save(out/'PLAN.json',plan)
    print('PREPARED2500train/valframes50sessions; no test payload access',flush=True)


def process_session(session):
    assert session['split'] in ('train','val')
    assert P.sha(DATA/session['scene']/'description.json')==session['description_sha256']
    g=S.camera_geometry(session['intrinsics'])
    frames=[]
    for item in session['rows']:
        path=DATA/item['depth']
        assert P.sha(path)==item['depth_sha256'],item['id']
        raw=S.decode_depth(path)
        assert tuple(raw.shape)==tuple(g['shape'])
        truth=S.truth_hypotheses(raw,g)
        needed=any(np.any(t['contact']) for t in truth.values())
        mask_path=DATA/item['rgb'].replace('/video_frames/','/segmentation_masks/')
        semantics=None
        mask_info=dict(required_for_known_contact=needed,status='NOT_NEEDED')
        if needed:
            if mask_path.is_file():
                with Image.open(mask_path) as image:
                    assert image.mode=='RGB'
                    semantics=np.asarray(image)[:,:,0]
                    assert semantics.shape==raw.shape
                mask_info.update(status='READ',sha256=P.sha(mask_path),path=str(mask_path.relative_to(ROOT)))
            else:
                mask_info['status']='MISSING'
        rows=[]
        for hypothesis,t in truth.items():
            depth=raw if hypothesis=='optical_Z' else raw/g['radial_factor']
            for i,q in enumerate(g['queries']):
                row={k:item[k] for k in ('id','scene','split')}
                row.update(camera=session['camera'],hypothesis=hypothesis,query=i,query_name=q['name'],
                    category=str(t['category'][i]),contact_bin=str(t['contact_bin'][i]),
                    truth={k:v[i] if isinstance(v,np.ndarray) and v.ndim and v.shape[0]==2 else v for k,v in t.items() if k!='query_names'},
                    contact_semantic_pixels=None,contact_semantic_supported_classes=None)
                if semantics is not None and t['contact'][i]:
                    take=g['fov_mask'] & np.isfinite(depth) & (depth>=.6) & (depth<2.1)
                    take &= (depth*g['fy']>=q['y_low']) & (depth*g['fy']<=q['y_high']) & (np.abs(depth*g['fx'])<.3)
                    counts=np.bincount(semantics[take],minlength=256)
                    assert int(counts.sum())==int(t['contact_count'][i])
                    row['contact_semantic_pixels']={str(j):int(counts[j]) for j in np.flatnonzero(counts)}
                    row['contact_semantic_supported_classes']=[int(j) for j in np.flatnonzero(counts>=16)]
                rows.append(P.plain(row))
        frames.append(dict(id=item['id'],mask=mask_info,rows=rows))
    return dict(scene=session['scene'],camera=session['camera'],split=session['split'],frames=frames,
        geometry=P.plain({k:v for k,v in g.items() if not isinstance(v,np.ndarray) or v.size<=1000}))


def summarize(rows,taxonomy):
    categories={}
    for stratum in P.STRATA:
        group=[r for r in rows if (r['contact_bin']==stratum if stratum.startswith('contact') and stratum!='contact' else r['category']==stratum)]
        semantic_counts=Counter()
        for r in group:
            semantic_counts.update(r['contact_semantic_supported_classes'] or [])
        categories[stratum]=dict(queries=len(group),frames=len({r['id'] for r in group}),sessions=len({r['scene'] for r in group}),
            contact_semantic_support_queries={taxonomy.get(str(k),str(k)):v for k,v in sorted(semantic_counts.items())})
    return dict(queries=len(rows),categories=categories)


def run(out,workers):
    if (out/'result.json').exists() or (out/'session-ledger.json').exists():
        raise FileExistsError('Preserve result')
    plan=P.read(out/'PLAN.json')
    for path,expected in plan['source_sha256'].items():
        assert P.sha(ROOT/path)==expected,path
    started=time.perf_counter();sessions=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,s in enumerate(pool.map(process_session,plan['sessions']),1):
            sessions.append(s)
            print('scored sessions',i,'/50; frames',i*50,flush=True)
    P.save(out/'session-ledger.json',sessions)
    frames=[f for s in sessions for f in s['frames']]
    rows=[r for f in frames for r in f['rows']]
    assert len(frames)==2500 and len(rows)==10000
    taxonomy={str(v):k for k,v in P.read(DATA/'official-taxonomy/labelmap.json').items()}
    result=dict(role=plan['role'],plan_sha256=P.sha(out/'PLAN.json'),frames=len(frames),queries_per_hypothesis=5000,
        sessions=len(sessions),camera_sessions=dict(Counter(s['camera'] for s in sessions)),
        mask_read_frames=sum(f['mask']['status']=='READ' for f in frames),
        mask_missing_frames=sum(f['mask']['status']=='MISSING' for f in frames),hypotheses={})
    for h in HYPOTHESES:
        population=[r for r in rows if r['hypothesis']==h]
        result['hypotheses'][h]=dict(all=summarize(population,taxonomy),
            by_split={sp:summarize([r for r in population if r['split']==sp],taxonomy) for sp in ('train','val')},
            by_camera={c:summarize([r for r in population if r['camera']==c],taxonomy) for c in ('camera_head','camera_chest')})
    pairs=defaultdict(dict)
    for r in rows:
        pairs[(r['id'],r['query'])][r['hypothesis']]=r
    transitions=Counter((v['optical_Z']['category'],v['radial']['category']) for v in pairs.values())
    result['depth_hypothesis_category_transitions']={f'{a}->{b}':n for (a,b),n in sorted(transitions.items())}
    result['depth_hypothesis_shallow_bin_transitions']=dict(Counter(
        v['optical_Z']['contact_bin']+'->'+v['radial']['contact_bin'] for v in pairs.values()
        if any(v[h]['contact_bin'] in ('contact0-2','contact2-5') for h in HYPOTHESES)))
    result['native_partial_frames']=sum(len(s['frames']) for s in sessions if not s['geometry']['native_covers_nominal_fov'])
    result['seconds']=time.perf_counter()-started
    result['verdict']='SUPPORT_AUDIT_COMPLETE_DEPTH_CONVENTION_UNRESOLVED'
    P.save(out/'result.json',result)
    lines=['# SANPO whole-query支持量：两种深度解释','',result['verdict'],'',
        '仅train2000/40session、val500/10session；未访问test像素；仍为已消费Development。',
        'HEAD/BODY只是相机相对虚拟查询名，非穿戴者身体外参。45°×45°公共视场，原生2208×1242、16像素支持。',
        '源契约尚未确定光轴Z/径向距离，两个解释全部保留。语义仅诊断接触点构成，不过滤或定义几何标签。','',
        '|深度解释|类别|query/n|帧|session|','|---|---|---|---|---|']
    for h,data in result['hypotheses'].items():
        for name,s in data['all']['categories'].items():
            lines.append(f"|{h}|{name}|{s['queries']}/5000|{s['frames']}|{s['sessions']}|")
    lines+=['','全部split/camera分层、语义支持类别、深度解释转移、UNKNOWN/FOV/遮挡分母在result与session-ledger保留。',
        '每session前50帧，不是独立障碍数量；物理布局独立性未知。像素支持不同于旧Hypersim分辨率，不直接比较发生率。',
        '没有RGB推理、训练、ToF合成、下载或任务选择；候选支持不证明RGB可识别或可融合。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(result['verdict'],result['seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('prepare','run'))
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    args=p.parse_args()
    if args.stage=='prepare':
        prepare(args.out)
    else:
        run(args.out,args.workers)
