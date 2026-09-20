'use strict';
// Engineering replay of sealed alert flags. No sensor inference or truth scoring.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '../../../../..');
const source = path.join(root, 'artifacts.local/work/ba-full-event-transfer-20260920');
const output = path.join(root, 'artifacts.local/work/ba-event-notification-20260920');
const schedulerFile = path.join(__dirname, 'event_notifications.js');
const arms = ['strongest_hold', 'closest_hold'];
const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const read = file => JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, ''));
const save = (name, value) => fs.writeFileSync(path.join(output, name), JSON.stringify(value, null, 2) + '\n', {flag: 'wx'});
let assertions = 0;
function same(actual, expected, reason) {
  assertions++;
  assert.deepEqual(actual, expected, reason);
}
function verifySeal(name) {
  const seal = read(path.join(source, name));
  same(seal.status, 'COMPLETE', name + ' complete');
  same(seal.frames, 576, name + ' frames');
  same(seal.protocol_sha256, digest(path.join(source, 'protocol.json')), name + ' protocol binding');
  for (const [file, hash] of Object.entries(seal.hashes)) {
    same(digest(path.join(source, file)), hash, name + ' payload ' + file);
  }
}
function sample(row, arm) {
  // Report grouping tags and evaluator truth never enter the scheduler.
  return {clipId: row.clip_id, frameIndex: row.frame_in_clip, timeS: row.time_s, alert: row.flags[arm]};
}
function group(clipId) {
  if (clipId.endsWith('_boundary')) return 'Boundary';
  assert.ok(clipId.endsWith('_inside') || clipId.endsWith('_outside'), 'Known report-only cohort suffix');
  return 'Core';
}
function expectedSegments(rows, arm) {
  const segments = [];
  let active = null;
  for (const row of rows) {
    if (active && (!row.flags[arm] || row.clip_id !== active.clip_id)) {
      active.stop = {id: row.id, frameIndex: row.frame_in_clip, timeS: row.time_s, clipId: row.clip_id};
      active = null;
    }
    if (row.flags[arm] && !active) {
      active = {clip_id: row.clip_id, start: {id: row.id, frameIndex: row.frame_in_clip, timeS: row.time_s, clipId: row.clip_id}, stop: null};
      segments.push(active);
    }
  }
  return segments;
}
function replay(Scheduler, rows, arm, mode, trace) {
  let context = null, active = null;
  const emissions = [], cancellations = [], issued = [];
  const scheduler = new Scheduler({
    emit(event) {
      same(active, null, arm + ' emits only after prior event ended');
      assert.ok(event && Object.hasOwn(event, 'id'), 'emit token identity');
      assertions++;
      active = event.id;
      same(issued.includes(event.id), false, 'no token reuse');
      issued.push(event.id);
      const record = {arm, mode, action: 'emit', token: event.id, ...context};
      emissions.push(record);
      trace.push(record);
    },
    cancel() {
      if (active !== null) {
        const record = {arm, mode, action: 'cancel', token: active, ...context};
        cancellations.push(record);
        trace.push(record);
      }
      active = null;
    }
  });
  for (const row of rows) {
    context = {id: row.id, clipId: row.clip_id, frameIndex: row.frame_in_clip, timeS: row.time_s};
    const oldToken = active, before = emissions.length;
    const value = sample(row, arm);
    const state = scheduler.step(value);
    same(active !== null, value.alert, arm + ' active state exactly follows sealed flag');
    if (oldToken !== null && oldToken !== active) {
      same(scheduler.consume(oldToken), false, 'ended event rejects late callback');
    }
    if (mode === 'consume_immediately' && emissions.length > before) {
      same(scheduler.consume(active), true, 'active event token consumed once');
      same(scheduler.consume(active), false, 'duplicate callback cannot speak twice');
    }
    const beforeDuplicate = [emissions.length, cancellations.length, active];
    const duplicateState = scheduler.step(value);
    same([emissions.length, cancellations.length, active], beforeDuplicate, 'duplicate frame causes no notification side effect');
    same(duplicateState, state, 'duplicate frame returns unchanged state');
    if (!value.alert) {
      // In leave_pending mode these callbacks were never consumed, so failure
      // would expose delayed speech after clear, rather than mere double use.
      for (const token of issued) same(scheduler.consume(token), false, 'all old callbacks invalid after false');
    }
  }
  context = {id: 'playback-reset', clipId: null, frameIndex: null, timeS: null};
  scheduler.reset();
  same(active, null, 'reset clears active event');
  for (const token of issued) same(scheduler.consume(token), false, 'reset invalidates old callback tokens');
  const expected = expectedSegments(rows, arm);
  same(emissions.map(e => ({id:e.id, clipId:e.clipId, frameIndex:e.frameIndex, timeS:e.timeS})),
       expected.map(e => e.start), arm + ' exact segment onset');
  same(cancellations.map(e => ({id:e.id, clipId:e.clipId, frameIndex:e.frameIndex, timeS:e.timeS})),
       expected.map(e => e.stop || {id:'playback-reset',clipId:null,frameIndex:null,timeS:null}), arm + ' exact segment stop');
  return {emissions, cancellations};
}
function lifecycleProbe(Scheduler, rows, trace) {
  const row = rows.find(r => r.flags.strongest_hold);
  assert.ok(row, 'Probe needs an actual sealed alert sample');
  let issued = [], cancellations = 0;
  const scheduler = new Scheduler({emit: event => issued.push(event.id), cancel: () => cancellations++});
  const value = sample(row, 'strongest_hold');
  scheduler.step(value);
  const pending = issued.at(-1);
  scheduler.reset();
  same(scheduler.consume(pending), false, 'pending callback invalid after playback reset while active');
  same(cancellations > 0, true, 'active playback reset calls cancel');
  scheduler.step(value);
  const spoken = issued.at(-1);
  same(scheduler.consume(spoken), true, 'new token after reset');
  const before = cancellations;
  scheduler.reset();
  same(cancellations > before, true, 'reset cancels even already spoken event');
  same(scheduler.consume(spoken), false, 'spoken token remains invalid');
  scheduler.step(value);
  const preSeek = issued.at(-1), emissionCount = issued.length;
  // Playback control probe: skip to another real sample in this sealed clip.
  const skipped = rows.find(r => r.clip_id === row.clip_id && r.frame_in_clip === row.frame_in_clip + 2);
  assert.ok(skipped, 'Actual sealed sample for seek probe');
  scheduler.step(sample(skipped, 'strongest_hold'));
  same(scheduler.consume(preSeek), false, 'discontinuous playback invalidates prior pending token');
  same(issued.length - emissionCount, skipped.flags.strongest_hold ? 1 : 0, 'seek starts current sample afresh without gap filling');
  scheduler.reset();
  trace.push({action:'lifecycle_probe',source_frame:row.id,seek_frame:skipped.id,
    checks:['active reset invalidates pending','reset cancels spoken event','seek invalidates prior token without gap fill']});
}

fs.mkdirSync(output, {recursive:true});
assert.ok(!fs.existsSync(path.join(output, 'summary.json')), 'Preserve completed replay receipt; do not overwrite');
const trace = [];
let result;
try {
  verifySeal('observation-seal.json');
  verifySeal('prediction-seal.json');
  const predictionsHash = digest(path.join(source, 'predictions.json'));
  const schedulerHash = digest(schedulerFile);
  const runnerHash = digest(__filename);
  const rows = read(path.join(source, 'predictions.json'));
  same(rows.length, 576, 'complete sealed row count');
  same(new Set(rows.map(r => r.id)).size, 576, 'unique sealed identities');
  const clips = new Map();
  for (const row of rows) {
    if (!clips.has(row.clip_id)) clips.set(row.clip_id, []);
    clips.get(row.clip_id).push(row);
    for (const arm of arms) same(typeof row.flags[arm], 'boolean', 'sealed alert boolean');
  }
  same(clips.size, 24, 'complete 24 clip set');
  for (const [id, seq] of clips) {
    same(seq.length, 24, id + ' frame count');
    same(seq.map(r => r.frame_in_clip), Array.from({length:24}, (_, i) => i), id + ' ordering');
    same(seq.every((r, i) => Math.abs(r.time_s - .2 * i) < 1e-8), true, id + ' timestamps');
  }
  const {Scheduler} = require(schedulerFile);
  same(typeof Scheduler, 'function', 'actual Scheduler export');
  const summaries = {};
  for (const arm of arms) {
    const immediate = replay(Scheduler, rows, arm, 'consume_immediately', trace);
    const pending = replay(Scheduler, rows, arm, 'leave_pending', trace);
    same(immediate.emissions.map(e=>e.id), pending.emissions.map(e=>e.id), 'pending timing does not change event selection');
    summaries[arm] = {};
    for (const scope of ['overall','Core','Boundary']) {
      const selected = rows.filter(r => scope === 'overall' || group(r.clip_id) === scope);
      const perFrameRequests = selected.filter(r=>r.flags[arm]).length;
      const emits = immediate.emissions.filter(e=>scope === 'overall' || group(e.clipId) === scope).length;
      summaries[arm][scope] = {frames:selected.length, clips:new Set(selected.map(r=>r.clip_id)).size,
        hypothetical_per_positive_frame_requests:perFrameRequests, actual_event_notification_requests:emits,
        requests_removed:perFrameRequests-emits,
        request_reduction_fraction:perFrameRequests ? (perFrameRequests-emits)/perFrameRequests : null};
    }
  }
  lifecycleProbe(Scheduler, rows, trace);
  same(digest(path.join(source, 'predictions.json')), predictionsHash, 'input unchanged after replay');
  same(digest(schedulerFile), schedulerHash, 'scheduler unchanged during replay');
  verifySeal('prediction-seal.json');
  result = {status:'PASS',scope:'ENGINEERING_ONLY_SEALED_ALERT_NOTIFICATION_REPLAY',assertions,
    source_frames:576,source_clips:24,replayed_arm_frames:1152,
    callback_delivery_modes:['consume_immediately','leave_pending'],
    hashes:{predictions:predictionsHash,prediction_seal:digest(path.join(source,'prediction-seal.json')),
      observation_seal:digest(path.join(source,'observation-seal.json')),scheduler:schedulerHash,runner:runnerHash},
    arms:summaries,
    checks:['one emit per contiguous true segment','exact onset and first-false cancellation',
      'no gap filling','duplicate sample ignored','active token consumed once',
      'late callbacks refused after false/reset/seek','reset cancels spoken event','sealed source unchanged'],
    limitations:['Request-count comparison uses a hypothetical request on every positive frame.',
      'No rendered or measured voice duration, loudness, human disturbance or real-time latency.',
      'No research re-evaluation: flags, threshold, hold and scientific gate remain frozen.',
      'Core/Boundary suffixes are used only for reporting; scheduler receives no truth or strata.']};
} catch (error) {
  result = {status:'FAIL',scope:'ENGINEERING_ONLY_SEALED_ALERT_NOTIFICATION_REPLAY',assertions,
    error:error.stack,runner_sha256:digest(__filename)};
}
result.completed_at_utc = new Date().toISOString();
save('trace.json', trace);
save('summary.json', result);
save('seal.json', {status:result.status,hashes:{'summary.json':digest(path.join(output,'summary.json')),
  'trace.json':digest(path.join(output,'trace.json'))}});
console.log(JSON.stringify(result,null,2));
if (result.status !== 'PASS') process.exitCode = 1;
