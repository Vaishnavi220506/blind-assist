/* Presentation contracts against the full exported replay, not a second scorer. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {stateFor, outcomeFor, flagFor, clipStats, dominantZone,hasTruth,evaluatedCohorts} = require('./app.js');
const dataPath = process.argv[2] || path.resolve(__dirname,
  '../../../../../artifacts.local/work/ba-core-alert-demo-20260920/site/demo-data.js');
const text = fs.readFileSync(dataPath, 'utf8').trim();
const data = JSON.parse(text.replace(/^window\.CORE_DEMO_DATA\s*=\s*/, '').replace(/;$/, ''));
let frameCount = 0;
for (const cohort of data.cohorts) {
  for (const clip of cohort.clips) for (const frame of clip.frames) {
    frameCount++;
    const state = stateFor(frame);
    assert.equal(state !== 'silent', frame.decision.alert);
    if (state === 'held') {
      assert.equal(frame.decision.strong, false);
      assert.equal(frame.decision.previous_strong, true);
    }
    // Evaluation labels affect TP/FP display but never the reminder category.
    const relabeled = {...frame, evaluation: {...frame.evaluation, truth: !frame.evaluation.truth}};
    assert.equal(stateFor(relabeled), state);
    for (const arm of ['calibration', 'strong', 'hold'])
      assert.equal(flagFor(relabeled, arm), flagFor(frame, arm));
    const zone = dominantZone(frame);
    assert.ok(zone >= 0 && zone < 64);
    const candidates = frame.zones.filter(z => z.possible);
    if (candidates.length) assert.equal(frame.zones[zone].joint, Math.max(...candidates.map(z => z.joint)));
  }
  for (const scope of ['all', 'core', 'boundary']) for (const arm of ['calibration', 'strong', 'hold']) {
    const clips = cohort.clips.filter(c => scope === 'all' || (scope === 'boundary') === (c.layout === 'BOUNDARY'));
    const total = clips.map(c => clipStats(c, arm)).reduce((a,b) =>
      Object.fromEntries(Object.keys(a).map(k => [k,a[k]+b[k]])), {tp:0,fp:0,fn:0,segments:0});
    const m = cohort.metrics[scope][arm];
    assert.deepEqual(total, {tp:m.TP,fp:m.FP,fn:m.FN,segments:m.false_segments}, `${cohort.id}/${scope}/${arm}`);
  }
}
assert.equal(frameCount, 1296);
assert.deepEqual([outcomeFor(true,true),outcomeFor(true,false),outcomeFor(false,true),outcomeFor(false,false)], ['TP','FN','FP','TN']);
// New variable-length illustrative clips preserve decisions without inventing negative truth.
const template=data.cohorts[0].clips[0].frames[0];
const unlabeled={...template,evaluation:{truth:null,boundary:null,relation:'UNLABELED'}};
assert.equal(hasTruth(unlabeled),false);
for(const alert of [true,false])assert.equal(outcomeFor(null,alert),'UNLABELED');
for(const arm of ['calibration','strong','hold']) {
  assert.equal(flagFor(unlabeled,arm),flagFor(template,arm));
  assert.deepEqual(clipStats({frames:Array(37).fill(unlabeled)},arm),{tp:0,fp:0,fn:0,segments:0});
}
assert.equal(stateFor(unlabeled),stateFor(template));
const fpFrame={...template,evaluation:{truth:false},decision:{alert:true}};
assert.deepEqual(clipStats({frames:[fpFrame,unlabeled,fpFrame]}),{tp:0,fp:2,fn:0,segments:2});
assert.deepEqual(evaluatedCohorts([...data.cohorts,{id:'showcase',illustrative:true,metrics:data.cohorts[0].metrics},{id:'no-metrics'}]),data.cohorts);
let illustrativeFrames=0;
const extraPath=path.join(path.dirname(dataPath),'extra-data.js');
if(fs.existsSync(extraPath)) {
  const extra=JSON.parse(fs.readFileSync(extraPath,'utf8').trim().replace(/^window\.CORE_DEMO_EXTRA\s*=\s*/,'').replace(/;$/,''));
  if(extra) {
    assert.equal(extra.illustrative,true);
    assert.equal(extra.metrics,undefined);
    assert.equal(extra.clips.length,8);
    assert.equal(new Set(extra.clips.map(c=>c.environment)).size,6);
    assert.deepEqual(evaluatedCohorts([...data.cohorts,extra]),data.cohorts);
    for(const clip of extra.clips) {
      assert.equal(clip.frames.length,24);
      assert.ok(clip.preview_index>=0&&clip.preview_index<clip.frames.length);
      assert.deepEqual(clipStats(clip),{tp:0,fp:0,fn:0,segments:0});
      for(const [i,frame] of clip.frames.entries()) {
        illustrativeFrames++;
        assert.equal(hasTruth(frame),false);
        assert.equal(outcomeFor(frame.evaluation.truth,frame.decision.alert),'UNLABELED');
        assert.equal(frame.decision.previous_strong,i>0?clip.frames[i-1].decision.strong:false);
        assert.equal(stateFor(frame)!=='silent',frame.decision.alert);
        assert.ok(Math.abs(frame.time_s-i*.2)<1e-6);
        assert.equal(frame.values.length,64);
        assert.equal(frame.zones.length,64);
        assert.deepEqual(frame.rgb_size,[1920,1080]);
        const image=fs.readFileSync(path.join(path.dirname(dataPath),frame.rgb));
        assert.equal(image.subarray(1,4).toString(),'PNG');
        assert.equal(image.readUInt32BE(16),1920);
        assert.equal(image.readUInt32BE(20),1080);
      }
    }
    assert.equal(illustrativeFrames,192);
  }
}
console.log(JSON.stringify({status:'PASS',frames:frameCount,illustrative_frames:illustrativeFrames,metric_groups:27,checks:['alert presentation','truth isolation','dominant zone','full metrics and FP segments','unlabeled dynamic clips','evaluation cohort isolation','native HD assets','clip-scoped temporal history']}));
