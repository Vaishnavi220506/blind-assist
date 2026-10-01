"""Descriptor-only crossfold AP guard agreed before evaluation; no re-fit."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[4] / 'artifacts.local/work/cnh-corridor-lofo-20260929'


def main():
    path = OUT / 'results.json'
    before = path.read_bytes()
    result = json.loads(before)
    backup = OUT / 'results-before-crossfold-ap-guard.json'
    if not backup.exists():
        backup.write_bytes(before)
    changed = 0
    for row in result['rows']:
        if row['family'] == 'pooled_crossfold':
            row['ap'] = None
            row['score_scale'] = 'fold_specific_unaligned'
            changed += 1
    result['crossfold_ap_guard'] = 'Pooling confusion counts is valid; raw scores from four distinct fold models are not aligned. AP is reported only within fold. Decision agreed with root before results; postprocessing implementation occurred after runner completed.'
    path.write_text(json.dumps(result, indent=2), encoding='utf8')
    receipt = dict(rows=changed, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   original_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),
                   final_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   unchanged='thresholds, predictions, confusion counts, bootstrap intervals, numerical gates')
    (OUT / 'crossfold-ap-guard.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
