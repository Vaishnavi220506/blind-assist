"""Assemble complete frozen episodes after mechanical interruption, never select outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    for key in ('source', 'prefix', 'resume', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    art = (Path(__file__).resolve().parents[5]/'artifacts.local').resolve()
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(art) or out == art:
        raise ValueError('Fresh canonical output required')
    source = json.loads(a.source.read_text())
    interrupted = json.loads((a.prefix/'interruption.json').read_text())
    receipt = json.loads((a.resume/'receipt.json').read_text())
    release = json.loads((a.resume/'process-release.json').read_text())
    assert receipt['status'] == 'PASS' and release['released']
    n = interrupted['retained_complete_frames']
    expected = [r['id'] for r in source['frames']]
    assert n % source['sampling']['steps_per_episode'] == 0
    assert receipt['frames'] == len(expected)-n
    assert sha(a.source) == interrupted['original_spec_sha256']
    assert sha(a.prefix/'spec.json') == sha(a.source)
    resumed_source = json.loads((a.resume/'spec.json').read_text())
    assert resumed_source['frames'] == source['frames'][n:]
    for folder in (a.prefix, a.resume):
        launch = json.loads((folder/'launch.json').read_text())
        assert launch['spec_sha256'] == sha(folder/'spec.json')
    # Compare capture dependencies: only the launcher/persistence plumbing changed.
    helpers = ('mz113_dynamic_sensors.py', 'mz115_zonal_sensors.py', 'mz115_zonal_tof.py',
               'mz99_angle_information_capture.py', 'ue_capture_readiness.py')
    assert all(sha(a.prefix/name) == sha(a.resume/name) for name in helpers)
    for name, digest in receipt['hashes'].items():
        assert sha(a.resume/name) == digest, name
    out.mkdir(parents=True)
    shutil.copyfile(a.source, out/'spec.json')
    hashes = {'spec.json': sha(out/'spec.json')}
    manifest = []
    for name in ('raw.jsonl', 'evaluator.jsonl', 'provenance.jsonl'):
        ids = []
        with (out/name).open('x', encoding='utf-8') as dest:
            for folder, limit in ((a.prefix, n), (a.resume, len(expected)-n)):
                with (folder/name).open(encoding='utf-8') as src:
                    for index, line in enumerate(src):
                        if index >= limit:
                            break
                        row = json.loads(line)
                        assert row['id'] == expected[len(ids)]
                        spec = source['frames'][len(ids)]
                        assert row['episode_id'] == spec['episode'] and row['time_s'] == spec['time_s']
                        if name == 'raw.jsonl':
                            path = Path(row['rgb_path'])
                            origin = (folder/path).resolve()
                            assert origin.is_relative_to(folder.resolve()) and not path.is_absolute()
                            target = out/path
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(origin, target)
                            assert sha(target) == sha(origin)
                            hashes[path.as_posix()] = sha(target)
                            manifest.append(dict(id=row['id'], source=str(folder.resolve()), rgb_path=path.as_posix()))
                        ids.append(row['id'])
                        dest.write(json.dumps(row, allow_nan=False)+'\n')
        assert ids == expected
        hashes[name] = sha(out/name)
    write(out/'manifest.json', manifest)
    hashes['manifest.json'] = sha(out/'manifest.json')
    write(out/'receipt.json', dict(status='PASS', frames=len(expected), episodes=source['episode_count'],
        authority='MECHANICALLY_ASSEMBLED_FROZEN_EPISODES_NOT_SINGLE_UNINTERRUPTED_CAPTURE',
        prefix_complete_frames=n, resume_frames=receipt['frames'],
        prefix=str(a.prefix.resolve()), resume=str(a.resume.resolve()),
        source_sha256=sha(a.source), spec_sha256=sha(a.source), hashes=hashes, predictions_used=False,
        prefix_limitation='Interrupted prefix has no engine PASS receipt; ordered complete streams and RGB validated during assembly',
        restart_rule='Only complete prefix episodes retained; incomplete episode replayed from original seed'))
    print(json.dumps(dict(status='PASS', frames=len(expected), output=str(out))))


if __name__ == '__main__':
    main()
