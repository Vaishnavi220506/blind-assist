const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {summary,history,transitions,supportVertices,volumeSVG}=require('./telemetry.js');
const site=path.resolve(__dirname,'../../../../../artifacts.local/work/ba-core-alert-demo-20260920/site');
const read=(name)=>JSON.parse(fs.readFileSync(path.join(site,name),'utf8').trim().replace(/^window\.\w+\s*=\s*/,'').replace(/;$/,''));
const data=read('demo-data.js'),extra=read('extra-data.js');
let count=0;
for(const co of [...data.cohorts,extra].filter(Boolean))for(const clip of co.clips){
  for(let i=0;i<clip.frames.length;i++){
    const f=clip.frames[i],snapshot=JSON.stringify(f),s=summary(f);
    assert.equal(s.valid,f.raw.valid_zones);
    assert.equal(s.state!=='silent',f.decision.alert);
    assert.deepEqual(summary({...f,evaluation:null}),s);
    assert.equal(history(clip,i).length,i+1);
    assert.ok(transitions(clip,i).every(e=>e.index<=i));
    const view=volumeSVG(f,data.geometry,35);
    assert.ok(!/NaN|Infinity|undefined/.test(view.svg));
    assert.equal(view.shown,f.zones.filter(z=>z.interval&&z.interval[0]<=4.2).length);
    assert.equal(JSON.stringify(f),snapshot);
    count++;
  }
  const future=new Proxy({}, {get(){throw Error('Future observation accessed');}});
  assert.equal(history({frames:[clip.frames[0],future]},0).length,1);
  assert.equal(transitions({frames:[clip.frames[0],future]},0).length,1);
}
assert.deepEqual(supportVertices({interval:null},{},4.2),[]);
assert.deepEqual(supportVertices({interval:[5,6]},{},4.2),[]);
const vertices=supportVertices({interval:[1,5]},{slopes_a:[-1,1],slopes_b:[0,2]},4);
assert.deepEqual(vertices[0],[-1,0,1]);
assert.deepEqual(vertices[6],[4,8,4]);
assert.equal(vertices.length,8);
console.log(JSON.stringify({status:'PASS',frames:count,checks:['observable-only summaries','future observation isolation','clip-local history','missing returns','support-frustum geometry','display clipping','immutable inputs','finite SVG']}));
