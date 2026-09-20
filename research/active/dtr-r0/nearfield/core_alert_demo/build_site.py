"""Assemble the offline replay site around the separately verified data export."""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
OUT=ROOT/'artifacts.local/work/ba-core-alert-demo-20260920'
SITE=OUT/'site'
FILES=('index.html','styles.css','app.js','launch_demo.cmd','serve_demo.py','README_使用说明.md')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

if __name__=='__main__':
    receipt=json.loads((OUT/'build-receipt.json').read_text(encoding='utf-8'))
    assert (SITE/'demo-data.js').is_file(),'Run build_data.py first'
    for name in FILES:shutil.copy2(HERE/name,SITE/name)
    (OUT/'site-build-receipt.json').write_text(json.dumps(dict(
        scope='Engineering replay packaging; no changed algorithm or new scientific evaluation',
        source_data_receipt_sha256=sha(OUT/'build-receipt.json'),
        files={n:sha(SITE/n) for n in FILES+('demo-data.js',)},
        offline=True,cdn_dependencies=0,frames=1296,clips=108),indent=2)+'\n',encoding='utf-8')
    print(SITE/'launch_demo.cmd')
