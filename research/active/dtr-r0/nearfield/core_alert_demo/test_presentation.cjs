/* Presentation contracts against the full exported replay, not a second scorer. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {stateFor, outcomeFor, flagFor, clipStats, dominantZone} = require('./app.js');
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
console.log(JSON.stringify({status:'PASS',frames:frameCount,metric_groups:27,checks:['alert presentation','truth isolation','dominant zone','full metrics and FP segments']}));
