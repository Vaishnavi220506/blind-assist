"""Verify independent entry equals frozen A* and one alert feeds all consumers."""
import hashlib
import json
from pathlib import Path
import numpy as np
from tolerance_eval import metric, temporal

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT/'artifacts.local/work/corridor-public-single-20260917'
HOME = ROOT/'artifacts.local/work/corridor-astar-effect-20260917'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def lines(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    pack = read(HOME/'bundle-receipt.json')
    for p, h in {**pack['originals'], **pack['sources']}.items():
        assert sha(p) == h, p
    for name, h in pack['bundle'].items():
        assert sha(HOME/'bundle'/name) == h
    replay = HOME/'replay'
    done = read(replay/'completion.json')
    assert done['status'] == 'PASS'
    for name, h in done['outputs'].items():
        assert sha(replay/name) == h, name
    seal = read(replay/'input-seal.json')
    for p, h in {**seal['inputs'], **seal['sources']}.items():
        assert sha(p) == h
    original = read(SOURCE/'confirmation/predictions.json')
    cases = read(SOURCE/'confirmation/cases.json')
    predictions, reminders = lines(replay/'predictions.jsonl'), lines(replay/'reminders.jsonl')
    rawroot = SOURCE/'source/returned-v1/capture-v1'
    raw = lines(rawroot/'raw.jsonl')
    receipt = read(rawroot/'receipt.json')
    for name, h in receipt['hashes'].items():
        assert sha(rawroot/name) == h
    assert len(original) == len(cases) == len(predictions) == len(reminders) == len(raw) == 288
    config = read(HOME/'bundle/config.json')
    episode, yaw, prior_alert = None, 0., False
    for old, p, r, c, public in zip(original, predictions, reminders, cases, raw):
        assert old['id'] == p['id'] == r['id'] == c['id'] == public['id']
        assert p['score'] == old['control_score']
        assert p['alert'] == old['control'] == r['alert']
        assert p['threshold'] == config['threshold'] and p['model_id'] == config['model_id']
        assert p['alert'] == (p['score'] >= config['threshold'])
        if p['episode_id'] != episode:
            episode, yaw, prior_alert = p['episode_id'], 0., False
        if public['imu_valid']:
            yaw += public['delta_yaw']
        assert p['yaw'] == yaw and r['reminder_onset'] == (p['alert'] and not prior_alert)
        prior_alert = p['alert']
        assert p['complete_ms'] >= p['algorithm_ms'] >= 0
        assert p['complete_ms'] >= p['rgb_decode_ms'] >= 0
    flags = np.array([p['alert'] for p in predictions])
    truth = np.array([c['truth'] for c in cases])
    states = np.array([c['stratum'] for c in cases])
    clear = states != 'boundary'
    summary = dict(clear=metric(truth[clear], flags[clear]), strict=metric(truth, flags),
        boundary=metric(truth[~clear], flags[~clear]), temporal=temporal(cases, states, flags),
        strict_temporal=temporal(cases, np.where(truth, 'positive', 'negative'), flags))
    old_summary = read(SOURCE/'confirmation/summary.json')['methods']['A_retrained']
    assert all(v == old_summary[k] for k, v in summary.items())
    runtime = read(replay/'summary.json')
    assert runtime['model_count'] == 1 and runtime['forbidden_modules'] == []
    assert not any(runtime[k] for k in ['old_A_executed', 'positive_branch_executed', 'evaluator_read'])
    assert runtime['frames'] == 288 and runtime['episodes'] == 48
    for name, stats in runtime['latency_ms'].items():
        values = np.array([p[name] for p in predictions])
        assert stats == dict(mean=float(values.mean()), p50=float(np.percentile(values, 50)),
            p95=float(np.percentile(values, 95)), observed_max=float(values.max()))
    result = dict(status='PASS', frames=288, episodes=48, frozen_A_star_scores_bitwise=True,
        alerts_and_all_event_times_identical=True, saved_alert_equals_reminder_alert=True,
        reminder_onsets_and_IMU_resets_verified=True, original_confirmation_preserved=True,
        one_model_no_old_A_no_positive_no_torch=True, prediction_sha256=sha(replay/'predictions.jsonl'),
        output_metrics={k: summary[k] for k in ['clear', 'strict', 'boundary']},
        core_events=[summary['temporal']['core_events_detected'], summary['temporal']['core_events']],
        strict_events=[summary['strict_temporal']['core_events_detected'], summary['strict_temporal']['core_events']],
        latency_ms=runtime['latency_ms']['complete_ms'])
    (HOME/'runtime-audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
