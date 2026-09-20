"""Assemble the offline replay site around the separately verified data export."""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
OUT=ROOT/'artifacts.local/work/ba-core-alert-demo-20260920'
SITE=OUT/'site'
FILES=('index.html','styles.css','showcase.css','telemetry.js','app.js','replay_images.js','event_notifications.js','launch_demo.cmd','serve_demo.py','README_使用说明.md',
       'CITY_SHOWCASE_DELIVERY_20260920.md','EVENT_NOTIFICATION_DELIVERY_20260920.md')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

if __name__=='__main__':
    receipt=json.loads((OUT/'build-receipt.json').read_text(encoding='utf-8'))
    assert (SITE/'demo-data.js').is_file(),'Run build_data.py first'
    assert sha(SITE/'demo-data.js')==receipt['export_sha256'],'Frozen replay export changed'
    extra_path=SITE/'extra-data.js'
    if not extra_path.exists():
        extra_path.write_text('window.CORE_DEMO_EXTRA=null;\n',encoding='utf-8')
    extra=json.loads(extra_path.read_text(encoding='utf-8').strip().removeprefix('window.CORE_DEMO_EXTRA=').removesuffix(';'))
    for name in FILES:shutil.copy2(HERE/name,SITE/name)
    # Keep HTML and its presentation assets in one version after a local rebuild.
    page=(SITE/'index.html').read_text(encoding='utf-8')
    for name in FILES:
        if name.endswith(('.css','.js')):
            page=page.replace(f'"{name}"',f'"{name}?v={sha(SITE/name)[:12]}"')
    (SITE/'index.html').write_text(page,encoding='utf-8')
    (OUT/'site-build-receipt.json').write_text(json.dumps(dict(
        scope='Engineering replay packaging; no changed algorithm or new scientific evaluation',
        source_data_receipt_sha256=sha(OUT/'build-receipt.json'),
        files={n:sha(SITE/n) for n in FILES+('demo-data.js','extra-data.js')},
        offline=True,cdn_dependencies=0,frozen_evaluation_frames=1296,frozen_evaluation_clips=108,
        illustrative_frames=sum(len(c['frames']) for c in extra['clips']) if extra else 0,
        illustrative_clips=len(extra['clips']) if extra else 0),indent=2)+'\n',encoding='utf-8')
    print(SITE/'launch_demo.cmd')
