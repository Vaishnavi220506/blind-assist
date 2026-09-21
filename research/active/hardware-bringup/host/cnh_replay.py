"""Build an offline CNH replay from saved scores; never runs the classifier."""
from __future__ import annotations

import argparse
import base64
import bisect
import hashlib
import json
from pathlib import Path
import numpy as np
from capture import validate_frame
from cnh_components import marker_windows, PHASES, sha
from manual_orientation_review import load_streams
from orientation_review import safe_source

HERE = Path(__file__).resolve().parent


def nearest_camera(cameras, stamps, stamp, limit_ns=500_000_000):
    i = bisect.bisect_left(stamps, stamp)
    options = cameras[max(0,i-1):i+1]
    if not options:
        return None
    chosen = min(options, key=lambda c:(abs(c['host_ns']-stamp), c['host_ns']))
    return chosen if abs(chosen['host_ns']-stamp) <= limit_ns else None


def saved_scores(report):
    """Missing scores stay absent, including all fit/calibration frames."""
    if [z['zone'] for z in report['zones']] != list(range(16)):
        raise ValueError('Expected all16 saved zones in original order')
    result = {}
    for zone in report['zones']:
        for phase, values in zone.get('phases', {}).items():
            for row in values['frames']:
                key = (zone['zone'], phase, row['seq'])
                if key in result:
                    raise ValueError('Duplicate saved score identity')
                result[key] = row
    return result


def safe_json(data):
    return json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')


def build(run, report_path, previous_path, capture_protocol):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    if report.get('schema') != 'hardware-bringup.cnh-late-echo.v1' or report.get('execution_review_good') is not True:
        raise ValueError('Replay requires a supported saved report with valid execution review')
    if (Path(report['run']).name != run.name or report['protocol']['source_run'] != run.name
            or report['previous_report_sha256'] != sha(previous_path)):
        raise ValueError('Saved report/run identity mismatch')
    marker_path = run.parent/'.dashboard-control'/f'{run.name}.manual.json'
    markers = json.loads(marker_path.read_text(encoding='utf-8'))
    if markers['protocol_sha256'] != sha(capture_protocol) or sha(marker_path) != report['provenance']['markers_sha256']:
        raise ValueError('Capture protocol/markers changed')
    for relative, key in [('camera/serial.bin','camera_serial_sha256'),('tof/raw.bin','tof_raw_sha256')]:
        if sha(run/relative) != report['provenance'][key]:
            raise ValueError('Raw source hash changed')
    windows = marker_windows(markers)
    provenance = {'sources':{},'exclusions':[],'insufficient_reasons':[]}
    rows, cameras = load_streams(run,provenance)
    if provenance['exclusions'] or provenance['insufficient_reasons']:
        raise ValueError('Source validation failed')
    for name in ('camera','tof'):
        if provenance['sources'][name]['sha256'] != report['provenance']['sources'][name]['sha256']:
            raise ValueError('Timestamped frame index changed')
    image_hashes = {r['filename']:r['sha256'] for r in
                    (json.loads(line) for line in (run/'camera/frames.jsonl').read_text(encoding='utf-8').splitlines())}
    scores = saved_scores(report)
    stamps = [c['host_ns'] for c in cameras]
    frames, used, identities = [], set(), set()
    for name in PHASES:
        start,end = windows[name]
        for record in rows:
            stamp = record['host_received_monotonic_ns']
            if not start <= stamp < end:
                continue
            sensor = record['sensor']
            seq = sensor['seq']
            if (name,seq) in identities:
                raise ValueError('Duplicate source identity')
            identities.add((name,seq))
            block = (stamp-start)//1_000_000_000
            derived = validate_frame(sensor)
            camera = nearest_camera(cameras,stamps,stamp)
            image, delta = None, None
            if camera:
                path = safe_source(run,camera['path'])
                payload = path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != image_hashes[path.name]:
                    raise ValueError('Selected camera JPEG hash changed')
                image = 'data:image/jpeg;base64,'+base64.b64encode(payload).decode('ascii')
                delta = (camera['host_ns']-stamp)/1_000_000
            frame_scores = []
            for z in range(16):
                key = (z,name,seq)
                value = scores.get(key)
                if value is not None:
                    if value['block'] != block or value['scalar_known'] != bool(derived['range_valid'][z]):
                        raise ValueError('Saved score metadata no longer matches source')
                    if name in PHASES[:2] and block != 3:
                        raise ValueError('Reference/calibration frame incorrectly scored')
                    used.add(key)
                elif report['zones'][z]['eligible'] and (name in PHASES[2:] or block == 3):
                    raise ValueError('Expected evaluation score missing')
                frame_scores.append(value)
            frames.append({'seq':seq,'elapsed_s':(stamp-markers['start_host_monotonic_ns'])/1_000_000_000,
                           'phase':name,'block':block,
                           'role':'evaluation' if name in PHASES[2:] else 'reference' if block < 2 else 'calibration' if block==2 else 'check',
                           'camera_data_url':image,'camera_delta_ms':delta,
                           'hist':derived['hist_normalized'],'distance_mm':sensor['distance_mm'],
                           'range_valid':derived['range_valid'],'target_status':sensor['target_status'],'scores':frame_scores})
    if used != scores.keys():
        raise ValueError('Saved scores not fully represented in replay')
    if {name:sum(f['phase']==name for f in frames) for name in PHASES} != report['phase_counts']:
        raise ValueError('Frame denominators changed')
    templates = [{'zone':z,**{name:np.median([f['hist'][z] for f in frames
                    if f['phase']==name and f['role']=='reference'],axis=0).tolist()
                    for name in PHASES[:2]}} for z in range(16)]
    return {'schema':'hardware-bringup.cnh-replay.v1','run_id':run.name,
            'report_sha256':sha(report_path),'protocol_sha256':report['protocol_sha256'],
            'protocol':report['protocol'],'execution_review_good':report['execution_review_good'],
            'zones':report['zones'],'templates':templates,'frames':frames,
            'limits':['已有记录的回放；未重新评分或新增实验证据。',
                      '主机接收时间就近配对（最多500 ms），非曝光同步；阶段间等待已跳过。',
                      '原始区域顺序，未做像素投影；颜色不是物体位置或告警。',
                      '近端与后峰是探索性波形证据，不是两个目标或距离恢复。',
                      '240/242 半遮挡区域帧涉及幅值外推；少量纯对照不能证明普遍低误报率。'],
            'provenance':report['provenance'], 'previous_report_sha256':sha(previous_path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','report','previous-report','capture-protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    data=build(args.run.resolve(),args.report.resolve(),args.previous_report.resolve(),args.capture_protocol.resolve())
    template=(HERE/'cnh_replay.html').read_text(encoding='utf-8')
    if template.count('__CNH_REPLAY_DATA__') != 1:
        raise ValueError('Invalid replay template')
    page=template.replace('__CNH_REPLAY_DATA__',safe_json(data))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'index.html').write_text(page,encoding='utf-8')
    manifest={'schema':data['schema'],'run_id':data['run_id'],'report_sha256':data['report_sha256'],
              'exporter_sha256':sha(__file__),'template_sha256':sha(HERE/'cnh_replay.html'),
              'page_sha256':sha(args.output/'index.html'),'frames':len(data['frames']),
              'scored_region_frames':sum(s is not None for f in data['frames'] for s in f['scores']),
              'paired_camera_frames':sum(f['camera_data_url'] is not None for f in data['frames']),
              'max_absolute_camera_delta_ms':max(abs(f['camera_delta_ms']) for f in data['frames'] if f['camera_delta_ms'] is not None),
              'score_source':'Saved report only; no classifier invocation'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest))


if __name__=='__main__':
    main()
