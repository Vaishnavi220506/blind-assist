const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const {model,projection,history}=require('./mechanism.js');
const site=path.resolve(__dirname,'../../../../../artifacts.local/work/ba-core-alert-demo-20260920/site');
const read=name=>JSON.parse(fs.readFileSync(path.join(site,name),'utf8').trim().replace(/^window\.\w+\s*=\s*/,'').replace(/;$/,''));
const data=read('demo-data.js'),extra=read('extra-data.js');
let frames=0,missing=0,depthDisjoint=0;
for(const co of [...data.cohorts,extra].filter(Boolean))for(const clip of co.clips){
 for(let i=0;i<clip.frames.length;i++){
  const frame=clip.frames[i],before=JSON.stringify(frame);
  // Any accidental use of evaluator labels must fail the display contract.
  const observable={...frame};Object.defineProperty(observable,'evaluation',{get(){throw Error('Evaluator data used');}});
  const m=model(observable,35,data.thresholds.strong);
  assert.equal(m.strong,!!(m.definite||(m.possible&&m.score>=data.thresholds.strong)));
  if(m.winner!==null){assert.equal(frame.zones[m.winner].possible,true);assert.equal(frame.zones[m.winner].joint,frame.score);}
  const absent=frame.values.findIndex(x=>x===null);
  if(absent>=0){const a=model(observable,absent,data.thresholds.strong);assert.equal(a.interval,null);assert.equal(a.angular,null);assert.equal(a.status,'无有效返回');missing++;}
  const disjoint=frame.zones.findIndex(z=>z.interval&&z.depth===0);
  if(disjoint>=0){assert.equal(model(observable,disjoint,data.thresholds.strong).status,'有效返回 · 无深度交集');depthDisjoint++;}
  const selected=m.winner??35;
  const svgs=[false,true].map(side=>projection(frame.zones[selected],data.geometry.zone_geometry[selected],data.geometry.volume,side));
  for(const svg of svgs){assert.ok(!/NaN|Infinity|undefined/.test(svg));for(const match of svg.matchAll(/points="([^"]+)"/g))for(const pair of match[1].split(' ')){const [x,y]=pair.split(',').map(Number);assert.ok(x>=0&&x<=320&&y>=0&&y<=248,'Support projection must fit its declared viewport');}}
  assert.ok(svgs[0].includes('id="mechanism-top"'));assert.ok(svgs[1].includes('id="mechanism-side"'));
  assert.equal(history(clip,i).length,i+1);assert.equal(JSON.stringify(frame),before);frames++;
 }
 const future=new Proxy({}, {get(){throw Error('Future observation used');}});
 assert.equal(history({frames:[clip.frames[0],future]},0).length,1);
}
assert.ok(missing>0&&depthDisjoint>0);
console.log(JSON.stringify({status:'PASS',frames,checks:['observable-only principle model','sealed strong route parity','eligible max-zone distinction','missing versus depth-disjoint returns','full support fits both projections','unique diagram IDs','causal history','immutable data']}));
