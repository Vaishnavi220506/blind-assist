"""Read-only paired diagnostics supplement, no threshold or model changes."""
import numpy as np
from run_bg_invariance import OUT, read, write, sha


def main():
    summary=read(OUT/'summary.json')
    result={}
    for dataset in ('old','new'):
        cases=read(OUT/f'{dataset}-cases.json')
        result[dataset]={}
        for arm in ('raw_hgb','A_star','B0','B1','B2'):
            bygroup={}
            for c in cases:
                if c['stratum']=='boundary':continue
                value=int(c['bg']['flags'][arm]==c['truth'])-int(c['bg']['flags']['raw_hgb']==c['truth'])
                bygroup[c['group']]=bygroup.get(c['group'],0)+value
            mm=summary['methods'][dataset][arm]
            result[dataset][arm]=dict(clear_group_wins=sum(v>0 for v in bygroup.values()),
                clear_group_ties=sum(v==0 for v in bygroup.values()),clear_group_losses=sum(v<0 for v in bygroup.values()),
                strict_lost_events=sum(c['baseline'] is not None and c['current'] is None for c in mm['strict_events_changes']),
                strict_gained_events=sum(c['baseline'] is None and c['current'] is not None for c in mm['strict_events_changes']),
                strict_delayed_events=sum(c['baseline'] is not None and c['current'] is not None and c['current']>c['baseline'] for c in mm['strict_events_changes']),
                core_event_changes=mm['core_changes'],native_support_changes=mm['changes']['native_supported'])
        if dataset=='new':
            # Pair texture variants by identical physical group, member and time.
            keyed={}
            for i,c in enumerate(cases):
                member=c['episode_id'].rsplit('_',1)[1]
                keyed.setdefault((c['group'],member,c['time_s']),[]).append(i)
            for arm in result[dataset]:
                family={}
                for key,ii in keyed.items():
                    assert len(ii)==2
                    a,b=(cases[i] for i in ii)
                    name=a['family'];entry=family.setdefault(name,dict(drift=[],flips=0,pairs=0))
                    entry['pairs']+=1
                    entry['drift'].append(abs(a['bg']['probabilities'][arm]-b['bg']['probabilities'][arm]))
                    entry['flips']+=int(a['bg']['flags'][arm]!=b['bg']['flags'][arm])
                result[dataset][arm]['background_by_family']={k:dict(pairs=v['pairs'],flips=v['flips'],
                    mean_probability_drift=float(np.mean(v['drift']))) for k,v in family.items()}
    write(OUT/'diagnosis.json',dict(summary_sha256=sha(OUT/'summary.json'),datasets=result))
    print('PASS paired family and event-identity diagnosis')


if __name__=='__main__':main()
