import argparse,json
from pathlib import Path
from mz178_current_frame import sha,write,ART
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--timelines',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 rows={x['id']:x for x in (json.loads(v) for v in (a.predictions/'predictions.jsonl').read_text().splitlines() if v)}; seq=[json.loads(x) for x in a.timelines.read_text().splitlines()]
 stats={m:{'TP':0,'FP':0,'FN':0,'TN':0,'recovered':0,'baseline_miss_recovered':0,'new_FP':0,'merged_only_FP':0,'baseline_lost':0,'events':0} for m in ('broad','union','ambiguous','oracle')}; broad_ids=set()
 for s in seq:
  for f in s['frames']:
   r=rows[f['id']]; truth=f['risk_truth']
   for m in stats:
    al=r['arms'][m]['alert']; key='TP' if truth is True and al else 'FN' if truth is True else 'FP' if truth is False and al else 'TN'; stats[m][key]+=1
    if f['predictions']['astar']['alert'] and not al: stats[m]['baseline_lost']+=1
    if truth is True and al and not f['predictions']['astar']['alert']: stats[m]['recovered']+=1
    if truth is True and al and not f['predictions']['astar']['alert']: stats[m]['baseline_miss_recovered']+=1
    if truth is False and al and not f['predictions']['astar']['alert']: stats[m]['new_FP']+=1
    if truth is False and al and r['arms'][m]['contributors'] and all(c.get('status') in ('NATIVE_COMPONENT','SOURCE_ORACLE') for c in r['arms'][m]['contributors']): stats[m]['merged_only_FP']+=1
  for m in stats:
   stats[m]['events']+=int(any(rows[f['id']]['arms'][m]['alert'] and f['risk_truth'] is True and f['time_s']<s['contact_first_s'] for f in s['frames']) if s['contact_first_s'] is not None else 0)
 for m,v in stats.items(): v['precision']=v['TP']/(v['TP']+v['FP']);v['recall']=v['TP']/(v['TP']+v['FN'])
 write(a.output,{'status':'PASS','arms':stats,'inputs':{'predictions':sha(a.predictions/'predictions.jsonl'),'timelines':sha(a.timelines)}})
if __name__=='__main__': main()
