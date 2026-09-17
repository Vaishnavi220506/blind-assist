"""Compact decision table and static research figure from sealed results only."""
from pathlib import Path
import json
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[5]
HOME=ROOT/'artifacts.local/work/corridor-public-single-20260917'
OUT=HOME/'confirmation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    summary=read(OUT/'summary.json');done=read(OUT/'completion.json')
    assert done['status']=='PASS' and sha(OUT/'summary.json')==done['summary_sha256']
    assert read(OUT/'independent-accounting.json')['status']=='PASS'
    assert read(OUT/'native-public-audit.json')['status']=='PASS'
    methods=summary['methods'];records={}
    for name,m in methods.items():
        records[name]=dict(clear=m['clear'],strict=m['strict'],boundary=m['boundary'],
            clear_changes=m['clear_changes'],strict_changes=m['strict_changes'],
            clear_negative_alert_fraction=m['temporal']['clear_negative_alert_fraction'],
            false_clear_episodes=m['temporal']['clear_negative_episodes_with_alert'],
            false_clear_segments=m['temporal']['clear_negative_alert_segments'],
            core_events=[m['temporal']['core_events_detected'],m['temporal']['core_events']],
            strict_events=[m['strict_temporal']['core_events_detected'],m['strict_temporal']['core_events']],
            core_onset_changes=m['first_core_changes'],strict_onset_changes=m['first_strict_changes'],
            changed_episodes=m['changed_episodes'],families=m['families'])
    cases=read(OUT/'cases.json');a=np.array([c['A'] for c in cases]);p=np.array([c['alert'] for c in cases]);r=np.array([c['control'] for c in cases])
    y=np.array([c['truth'] for c in cases]);clear=np.array([c['stratum']!='boundary' for c in cases]);w=np.array([c['sampled_witness'] for c in cases])
    opportunities=summary['opportunities']
    differences=dict(public_supported_clear_rescues=int(sum(clear&y&~a&p&w)),
        public_unsupported_clear_rescues=int(sum(clear&y&~a&p&~w)),
        public_remaining_clear_FN_with_support=int(sum(clear&y&~p&w)),
        public_remaining_clear_FN_without_support=int(sum(clear&y&~p&~w)),
        control_lost_A_clear_TP=int(sum(clear&y&a&~r)),control_new_clear_TP=int(sum(clear&y&~a&r)),
        control_removed_A_clear_FP=int(sum(clear&~y&a&~r)),control_new_clear_FP=int(sum(clear&~y&~a&r)))
    configurations=[]
    for group in sorted({c['group'] for c in cases}):
        mask=np.array([c['group']==group for c in cases])
        entry=dict(group=group,family=next(c['family'] for c in cases if c['group']==group),
            frames=int(mask.sum()),clear_frames=int(sum(mask&clear)))
        for label,flags in [('A',a),('A_plus_public',p),('A_retrained',r)]:
            entry[label]=dict(strict_correct=int(sum(mask&(flags==y))),
                clear_correct=int(sum(mask&clear&(flags==y))),
                clear_FP=int(sum(mask&clear&~y&flags)),clear_FN=int(sum(mask&clear&y&~flags)))
        configurations.append(entry)
    result=dict(methods=records,opportunities=opportunities,differences=differences,configurations=configurations,
        latency_ms=summary['latency_ms'],model_load_seconds=summary['model_load_seconds'],
        summary_sha256=sha(OUT/'summary.json'))
    (OUT/'decision-table.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['A','A_plus_public','A_retrained'];labels=['Frozen A','A + public positive','Same-data A*']
    colors=['#64748b','#0f766e','#d97706']
    fig,axs=plt.subplots(1,3,figsize=(12.2,3.6),layout='constrained')
    x=np.arange(3)
    for ax,key,title in zip(axs,['f1','recall','precision'],['Clear-task F1','Clear-task recall','Clear-task precision']):
        vals=[100*methods[n]['clear'][key] for n in names]
        ax.bar(x,vals,color=colors,width=.62)
        for i,val in enumerate(vals):ax.text(i,val+.7,f'{val:.2f}%',ha='center',fontsize=10)
        ax.set_xticks(x,labels,rotation=15,ha='right');ax.set_ylim(0,108);ax.set_title(title);ax.set_ylabel('%');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('One frozen model per method | 24 new configurations | 216 clear / 288 total frames',fontsize=12)
    fig.savefig(OUT/'single-confirmation.png',dpi=180);fig.savefig(OUT/'single-confirmation.svg');plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
