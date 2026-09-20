const assert=require('node:assert/strict');
const {Scheduler,Tone}=require('./event_notifications.js');
const sample=(i,alert,clipId='a')=>({clipId,frameIndex:i,timeS:i*.2,alert});
async function main(){
  const emitted=[];let cancels=0;
  const s=new Scheduler({emit:e=>emitted.push(e),cancel:()=>cancels++});
  [false,true,true,true,false,true].forEach((a,i)=>s.step(sample(i,a)));
  assert.deepEqual(emitted.map(e=>e.frameIndex),[1,5]);
  assert.equal(s.consume(emitted[0].id),false);assert.equal(s.consume(emitted[1].id),true);
  assert.equal(s.consume(emitted[1].id),false);
  s.step(sample(5,true));assert.equal(emitted.length,2);
  s.silence();s.step(sample(5,true));s.step(sample(6,true));assert.equal(emitted.length,2);
  s.step(sample(9,true));assert.equal(emitted.length,3); // Missing samples start a new session.
  s.step(sample(0,true,'b'));assert.equal(emitted.length,4);
  s.reset();assert.equal(s.consume(emitted.at(-1).id),false);
  s.step(sample(0,true,'b'));assert.equal(emitted.length,5);
  assert.throws(()=>s.step({...sample(1,false),alert:null}),TypeError);
  assert.equal(s.active,null);assert.ok(cancels>=5);

  let resolveResume,started=0,stopped=0,disconnected=0,closed=0;
  const context={currentTime:0,destination:{},resume:()=>Promise.resolve(),close:()=>{closed++;return Promise.resolve();},
    createOscillator:()=>({frequency:{},connect(){},disconnect(){disconnected++;},start(){started++;},stop(){stopped++;}}),
    createGain:()=>({gain:{setValueAtTime(){},linearRampToValueAtTime(){}},connect(){},disconnect(){}})};
  const tone=new Tone({makeContext:()=>context});await tone.enable();
  let promise;
  const q=new Scheduler({emit:e=>{promise=tone.notify(e,q);},cancel:()=>tone.cancel()});
  context.resume=()=>new Promise(r=>{resolveResume=r;});
  q.step(sample(0,true));q.step(sample(1,false));resolveResume();await promise;
  assert.equal(started,0,'Expired asynchronous audio request must not play');
  q.step(sample(2,true));q.reset();resolveResume();await promise;assert.equal(started,0);
  q.step(sample(3,true));tone.disable();resolveResume();await promise;assert.equal(started,0);
  context.resume=()=>Promise.resolve();await tone.enable();q.reset();q.step(sample(0,true));await promise;
  assert.equal(started,1);q.step(sample(1,true));assert.equal(started,1);
  q.step(sample(2,false));assert.equal(stopped,2);assert.equal(disconnected,1);assert.equal(tone.nodes.size,0);
  tone.close();assert.equal(closed,1);
  const broken=new Tone({makeContext:()=>{throw Error('no audio');}});assert.equal(await broken.enable(),false);
  console.log(JSON.stringify({status:'PASS',checks:['one request per predicted segment','duplicate suppression','pause continuity','seek and clip reset','invalid input cancellation','single token consumption','late resume after exit/reset/mute','active tone cancellation','audio unavailable','resource close']}));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
