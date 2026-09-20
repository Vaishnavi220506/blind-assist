/* Output-only notification scheduling. Never accepts truth or changes detector flags. */
(function(root){
  'use strict';
  class Scheduler {
    constructor({emit=()=>{},cancel=()=>{}}={}) {
      this.emit=emit;this.cancel=cancel;this.serial=0;this.last=null;this.active=null;this.pending=null;
    }
    silence(){this.pending=null;this.cancel();}
    reset(){this.silence();this.last=null;this.active=null;}
    consume(id){if(this.pending!==id||this.active!==id)return false;this.pending=null;return true;}
    step(sample){
      const {clipId,frameIndex,timeS,alert}=sample;
      if(typeof clipId!=='string'||!Number.isInteger(frameIndex)||!Number.isFinite(timeS)||typeof alert!=='boolean') {
        this.reset();throw new TypeError('Expected clip, integer frame, finite time and boolean alert');
      }
      const prev=this.last;
      if(prev&&prev.clipId===clipId&&prev.frameIndex===frameIndex&&prev.timeS===timeS&&prev.alert===alert)
        return {active:this.active!==null,id:this.active};
      if(prev&&(prev.clipId!==clipId||frameIndex!==prev.frameIndex+1||Math.abs(timeS-prev.timeS-.2)>1e-6))this.reset();
      this.last={clipId,frameIndex,timeS,alert};
      if(!alert&&this.active!==null){this.active=null;this.silence();}
      if(alert&&this.active===null){
        const id=++this.serial;this.active=id;this.pending=id;
        this.emit({id,clipId,frameIndex,timeS});
      }
      return {active:this.active!==null,id:this.active};
    }
  }
  // A local 120 ms tone, with no speech/network service or accumulating queue.
  // AudioContext.resume may finish late: consume() rejects expired event tokens.
  class Tone {
    constructor({makeContext,onStatus=()=>{}}){this.makeContext=makeContext;this.onStatus=onStatus;this.context=null;this.nodes=new Set();this.enabled=false;}
    async enable(){
      this.enabled=true;
      try{if(!this.context)this.context=this.makeContext();await this.context.resume();return this.enabled;}
      catch(_){this.enabled=false;this.onStatus('提示音不可用');return false;}
    }
    cancel(){for(const node of this.nodes){try{node.stop();}catch(_){}node.disconnect();}this.nodes.clear();}
    disable(){this.enabled=false;this.cancel();}
    async notify(event,scheduler){
      if(!this.enabled)return;
      try{
        await this.context.resume();
        if(!this.enabled||!scheduler.consume(event.id))return;
        const ctx=this.context,node=ctx.createOscillator(),gain=ctx.createGain(),now=ctx.currentTime;
        node.frequency.value=660;gain.gain.setValueAtTime(0,now);
        gain.gain.linearRampToValueAtTime(.08,now+.01);gain.gain.linearRampToValueAtTime(0,now+.12);
        node.connect(gain);gain.connect(ctx.destination);this.nodes.add(node);
        node.onended=()=>{this.nodes.delete(node);node.disconnect();gain.disconnect();};
        node.start(now);node.stop(now+.12);this.onStatus('已发出本段提示音');
      }catch(_){this.disable();this.onStatus('提示音不可用');}
    }
    close(){this.disable();if(this.context)this.context.close().catch(()=>{});}
  }
  const api={Scheduler,Tone};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.EventNotifications=api;
})(typeof window==='undefined'?{}:window);
