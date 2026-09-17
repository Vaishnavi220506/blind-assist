"""Copy the retained A* checkpoint into a one-model runtime bundle; no fit."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = ROOT/'artifacts.local/work/corridor-public-single-20260917'
OUT = ROOT/'artifacts.local/work/corridor-astar-effect-20260917'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = OUT/'bundle'
    assert not bundle.exists(), 'Preserve the existing effect bundle'
    config = json.loads((SOURCE/'bundle/config.json').read_text())
    checkpoint = SOURCE/'bundle/A-retrained.pkl'
    expected = config['model_hashes']['A-retrained.pkl']
    assert sha(checkpoint) == expected == 'dffd5c5a2d6d59c5a0546513148b21736d5aa8601e5a034bcdab9337295cd59c'
    assert config['control_threshold'] == .5568065433174727
    originals = {str(p): sha(p) for p in (SOURCE/'confirmation').iterdir() if p.is_file()}
    recipe = json.loads((SOURCE/'recipe-freeze.json').read_text())
    originals.update(recipe['bundle_files'])
    originals.update(recipe['sources'])
    assert all(sha(p) == h for p, h in originals.items())
    bundle.mkdir()
    shutil.copyfile(checkpoint, bundle/'A-star.pkl')
    write(bundle/'config.json', dict(schema='BLINDASSIST_STANDALONE_ASTAR_V1',
        model_id='A_STAR_1344_SEED185017', model_file='A-star.pkl', model_sha256=expected,
        threshold=config['control_threshold'], features=2485, models_loaded=1,
        public_inputs='RGB + 64-zone dual-return ToF + Radar + causal IMU',
        output='alert = A* score >= frozen threshold; zero ToF does not force a negative',
        source_recipe_sha256=sha(SOURCE/'recipe-freeze.json'), no_training=True))
    sources = {str(p): sha(p) for p in [Path(__file__), HERE/'astar_inference.py', HERE/'replay_astar.py']}
    write(OUT/'bundle-receipt.json', dict(status='PASS', originals=originals, sources=sources,
        bundle={p.name: sha(p) for p in bundle.iterdir()}, no_old_A_or_positive_weights=True,
        authority='ENGINEERING_EXTRACTION_OF_EXISTING_FROZEN_MODEL_NOT_NEW_EXPERIMENT'))
    print(json.dumps(dict(status='PASS', bundle=str(bundle), threshold=config['control_threshold'])))


if __name__ == '__main__':
    main()
