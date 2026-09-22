/* Display-only instruments. Every sample comes from the committed replay frame. */
(function(root){
  'use strict';
  const C={ink:'#0b141f',grid:'#233443',muted:'#849cac',text:'#dce9ed',teal:'#67e2c4',gold:'#e2bb79',blue:'#7fabe2'};
  const mode=f=>f.decision.held_only?'hold':f.decision.strong?'strong':'silent';
  const names={hold:'一帧保持',strong:'当前强证据',silent:'未触发提醒'};
  function summary(f){return {valid:f.values.filter(Number.isFinite).length,possible:f.zones.filter(z=>z.possible).length,
    definite:f.zones.filter(z=>z.definite).length,score:f.score,state:mode(f)};}
  function history(clip,index){return clip.frames.slice(0,index+1).map((f,i)=>({index:i,time:f.time_s,...summary(f)}));}
  function transitions(clip,index){return history(clip,index).filter((s,i,a)=>!i||s.state!==a[i-1].state).slice(-3);}
  function supportVertices(zone,geometry,far){
    if(!zone.interval||zone.interval[0]>far)return [];
    const [lo,upper]=zone.interval,hi=Math.min(upper,far),[a,b]=geometry.slopes_a,[c,d]=geometry.slopes_b;
    return [[a*lo,c*lo,lo],[b*lo,c*lo,lo],[b*lo,d*lo,lo],[a*lo,d*lo,lo],
      [a*hi,c*hi,hi],[b*hi,c*hi,hi],[b*hi,d*hi,hi],[a*hi,d*hi,hi]];
  }
  const EDGES=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  const FACE=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]];
  function volumeSVG(frame,geometry,selected,angle=35,far=4.2){
    const a=angle*Math.PI/180,scale=96*4.2/far;
    const project=([x,y,z])=>[260+scale*(Math.cos(a)*x+Math.sin(a)*(z-far/2)),155+scale*(.85*y-.45*(Math.cos(a)*(z-far/2)-Math.sin(a)*x))];
    const pt=p=>project(p).map(v=>v.toFixed(1)).join(',');
    const line=(p,q,color,width=1,extra='')=>`<line x1="${project(p)[0]}" y1="${project(p)[1]}" x2="${project(q)[0]}" y2="${project(q)[1]}" stroke="${color}" stroke-width="${width}" ${extra}/>`;
    const label=(p,text,color=C.muted)=>`<text x="${project(p)[0]}" y="${project(p)[1]}" fill="${color}" font-size="12">${text}</text>`;
    const wire=(v,color,width=1)=>EDGES.map(([i,j])=>line(v[i],v[j],color,width)).join('');
    let svg='<defs><clipPath id="support-view-clip"><rect x="8" y="8" width="504" height="286" rx="8"/></clipPath></defs><g clip-path="url(#support-view-clip)">';
    for(let z=0;z<=far;z+=1){svg+=line([-1.6,.9,z],[1.6,.9,z],C.grid,.7);svg+=label([1.66,.9,z],z+'m');}
    for(let x=-1.5;x<=1.5;x+=.5)svg+=line([x,.9,0],[x,.9,far],C.grid,.7);
    const {x,y,z}=geometry.volume,v=[[x[0],y[0],z[0]],[x[1],y[0],z[0]],[x[1],y[1],z[0]],[x[0],y[1],z[0]],
      [x[0],y[0],z[1]],[x[1],y[0],z[1]],[x[1],y[1],z[1]],[x[0],y[1],z[1]]];
    svg+=FACE.map(face=>`<polygon points="${face.map(i=>pt(v[i])).join(' ')}" fill="${C.teal}" fill-opacity=".028"/>`).join('');
    const zones=frame.zones.map((s,i)=>({s,i,v:supportVertices(s,geometry.zone_geometry[i],far)})).filter(q=>q.v.length);
    zones.sort((p,q)=>p.i===selected?1:q.i===selected?-1:q.s.interval[0]-p.s.interval[0]);
    for(const q of zones){
      const chosen=q.i===selected,color=chosen?C.gold:q.s.definite?C.teal:q.s.possible?'#a7bcae':'#536c81';
      svg+=`<g data-support-zone="${q.i}" role="button" aria-label="选择区域 ${q.i}，分数 ${q.s.joint.toFixed(4)}"><title>z${q.i} · ${q.s.interval.map(n=>n.toFixed(2)).join('–')} m</title>`;
      svg+=FACE.slice(0,4).map(face=>`<polygon points="${face.map(i=>pt(q.v[i])).join(' ')}" fill="${color}" fill-opacity="${chosen ? .1 : .025}"/>`).join('');
      svg+=wire(q.v,color,chosen?1.5:.5)+'</g>';
    }
    svg+=wire(v,C.teal,1.2)+label([.35,-.32,3],'固定通道',C.teal);
    const cam=project([0,0,0]);svg+=`<circle cx="${cam[0]}" cy="${cam[1]}" r="4" fill="${C.text}"/>`+label([.08,0,0],'相机',C.text);
    svg+=line([0,0,0],[0,0,far],C.teal,.7,'stroke-dasharray="3 5"')+label([0,-.18,far],'Z 前向');
    svg+='</g><text x="20" y="300" fill="#91aebc" font-size="11">相机坐标 · X 右 / Y 下 / Z 前</text>';
    return {svg,shown:zones.length};
  }
  if(typeof module!=='undefined'&&module.exports){module.exports={summary,history,transitions,supportVertices,volumeSVG};return;}
  const $=id=>document.getElementById(id);
  function canvas(id){
    const el=$(id),r=el.getBoundingClientRect();if(!r.width)return null;
    const w=r.width,h=r.height,dpr=Math.min(window.devicePixelRatio||1,2);
    if(el.width!==Math.round(w*dpr)||el.height!==Math.round(h*dpr)){el.width=Math.round(w*dpr);el.height=Math.round(h*dpr);}
    const ctx=el.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.font='12px system-ui';ctx.lineWidth=1;return {ctx,w,h};
  }
  class Instruments {
    constructor({geometry,threshold,onZone,onSeek}){
      Object.assign(this,{geometry,threshold,onZone,onSeek});this.current=null;this.pending=null;
      $('volumeAngle').oninput=()=>this.drawVolume();$('volumeRange').onchange=()=>this.drawVolume();
      $('historyMode').onchange=()=>this.drawHistory();
      $('supportVolume').onclick=e=>{const target=e.target.closest('[data-support-zone]');if(target)this.onZone(Number(target.dataset.supportZone));};
      $('historyZone').oninput=e=>this.onZone(Number(e.target.value));
      $('zoneHistory').onclick=e=>{
        if(!this.current||!this.historyBounds)return;const {index}=this.current,b=this.historyBounds,r=e.currentTarget.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
        const col=Math.floor((x-b.left)/b.cw),zone=Math.floor((y-b.top)/b.rh);
        if(col>=0&&col<=index&&zone>=0&&zone<64){this.onZone(zone);this.onSeek(col);}
      };
      this.observer=new ResizeObserver(()=>{cancelAnimationFrame(this.pending);this.pending=requestAnimationFrame(()=>this.drawCharts());});
      this.observer.observe($('zoneHistory'));this.observer.observe($('causalTrend'));
    }
    status(playing){$('sessionState').textContent=playing?'● 正在回放':'Ⅱ 回放已暂停';$('sessionState').classList.toggle('running',playing);}
    render(clip,index,zone){
      this.current={clip,index,zone};const f=clip.frames[index],s=summary(f);
      $('sessionClock').textContent=f.time_s.toFixed(1).padStart(4,'0')+' s';$('sessionFrame').textContent=`${f.id} · ${index+1} / ${clip.frames.length}`;
      $('instrumentValid').textContent=s.valid+' / 64';$('instrumentMissing').textContent=(64-s.valid)+' 区无有效返回';
      $('instrumentSupport').textContent=s.possible+' 区';$('instrumentDefinite').textContent=s.definite+' 区确定支持';
      $('instrumentScore').textContent=s.score.toFixed(3);$('instrumentState').textContent=names[s.state];
      $('instrumentState').dataset.state=s.state;$('instrumentHistory').textContent=f.decision.previous_strong?'前一采样：真实强证据':'前一采样：无可用强证据';
      $('historyZone').value=zone;this.drawVolume();this.drawCharts();
      $('eventReadout').innerHTML=transitions(clip,index).map(s=>`<button class="event-item ${s.state}" data-event-frame="${s.index}"><span>${s.time.toFixed(1)} s</span><b>${names[s.state]}</b><i>↗</i></button>`).join('');
      $('eventReadout').querySelectorAll('[data-event-frame]').forEach(b=>b.onclick=()=>this.onSeek(Number(b.dataset.eventFrame)));
    }
    drawVolume(){
      if(!this.current)return;const {clip,index,zone}=this.current,far=Number($('volumeRange').value),f=clip.frames[index];
      const view=volumeSVG(f,this.geometry,zone,Number($('volumeAngle').value),far);$('supportVolume').innerHTML=view.svg;
      const a=f.zones[zone];$('volumeCaption').textContent=`z${zone} · ${a.interval?a.interval.map(n=>n.toFixed(2)).join('–')+' m':'无有效返回'} · 显示 ${view.shown}/64 区（${far} m视窗）`;
    }
    drawCharts(){if(!this.current)return;this.drawHistory();this.drawTrend();}
    drawHistory(){
      const surface=canvas('zoneHistory');if(!surface||!this.current)return;
      const {ctx,w,h}=surface,{clip,index,zone}=this.current,left=30,top=20,pw=w-left-16,ph=h-53,cw=pw/clip.frames.length,rh=ph/64,mode=$('historyMode').value;
      this.historyBounds={left,top,cw,rh};ctx.fillStyle=C.ink;ctx.fillRect(left,top,pw,ph);
      for(let col=0;col<=index;col++)for(let z=0;z<64;z++){
        const f=clip.frames[col],value=f.values[z],x=left+col*cw,y=top+z*rh;
        if(value===null){ctx.strokeStyle='#344756';ctx.beginPath();ctx.moveTo(x,y+rh);ctx.lineTo(x+cw,y);ctx.stroke();continue;}
        const n=mode==='score'?f.zones[z].joint:1-Math.min(8,Math.max(0,value))/8;
        ctx.fillStyle=mode==='score'?`hsl(${190-150*n} 42% ${12+48*n}%)`:`hsl(${216-52*n} ${30+24*n}% ${16+33*n}%)`;
        ctx.fillRect(x+.3,y+.15,Math.max(.5,cw-.6),Math.max(.5,rh-.3));
      }
      ctx.fillStyle=C.muted;ctx.textAlign='right';
      for(let z=0;z<64;z+=8){ctx.fillText(String(z),left-6,top+z*rh+8);ctx.strokeStyle=C.grid;ctx.beginPath();ctx.moveTo(left,top+z*rh);ctx.lineTo(w-16,top+z*rh);ctx.stroke();}
      ctx.strokeStyle=C.gold;ctx.strokeRect(left,top+zone*rh,pw,rh);ctx.strokeStyle=C.text;ctx.strokeRect(left+index*cw,top,cw,ph);
      ctx.textAlign='center';for(let i=0;i<clip.frames.length;i+=Math.max(1,Math.ceil(clip.frames.length/5)))ctx.fillText(clip.frames[i].time_s.toFixed(1)+'s',left+(i+.5)*cw,h-12);
      ctx.textAlign='left';ctx.fillText('区',4,12);
      $('historyCaption').textContent=`z${zone} · 历史截至 ${clip.frames[index].time_s.toFixed(1)} s · ${mode==='score'?'分数 0 → 1':'近青绿 → 远深蓝'}；斜纹为缺失，右侧空白尚未播放。`;
      $('zoneHistory').setAttribute('aria-label',`64区${mode==='score'?'几何分数':'测距'}历史，截至${clip.frames[index].time_s.toFixed(1)}秒，选中区域${zone}`);
    }
    drawTrend(){
      const surface=canvas('causalTrend');if(!surface||!this.current)return;
      const {ctx,w,h}=surface,{clip,index}=this.current,samples=history(clip,index),left=32,right=16,top=22,bottom=29,pw=w-left-right,ph=h-top-bottom;
      const x=i=>left+i/Math.max(1,clip.frames.length-1)*pw,y=n=>top+(1-n)*ph;
      ctx.fillStyle=C.muted;ctx.textAlign='right';for(const n of [0,.25,.5,.75,1]){ctx.strokeStyle=C.grid;ctx.beginPath();ctx.moveTo(left,y(n));ctx.lineTo(w-right,y(n));ctx.stroke();if(n===0||n===.5||n===1)ctx.fillText(n.toFixed(1),left-7,y(n)+4);}
      ctx.strokeStyle=C.gold;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(left,y(this.threshold));ctx.lineTo(w-right,y(this.threshold));ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle=C.gold;ctx.textAlign='right';ctx.fillText('T '+this.threshold.toFixed(3),w-right,y(this.threshold)-7);
      // The fill follows the same discrete samples; no smoothing or synthesized observations.
      const wash=ctx.createLinearGradient(0,top,0,h-bottom);wash.addColorStop(0,'#67e2c438');wash.addColorStop(1,'#67e2c402');
      ctx.fillStyle=wash;ctx.beginPath();ctx.moveTo(x(0),y(0));samples.forEach((s,i)=>ctx.lineTo(x(i),y(s.score)));ctx.lineTo(x(index),y(0));ctx.closePath();ctx.fill();
      if(index<clip.frames.length-3){ctx.fillStyle='#9eb1bf';ctx.textAlign='center';ctx.fillText('待回放',(x(index)+w-right)/2,top+15);}
      for(const [key,divisor,color] of [['valid',64,C.blue],['possible',64,C.gold],['score',1,C.teal]]){
        ctx.strokeStyle=color;ctx.lineWidth=key==='score'?2.5:1.4;ctx.beginPath();samples.forEach((s,i)=>{if(i)ctx.lineTo(x(i),y(s[key]/divisor));else ctx.moveTo(x(i),y(s[key]/divisor));});ctx.stroke();
        const last=samples.at(-1);ctx.fillStyle=color;ctx.beginPath();ctx.arc(x(index),y(last[key]/divisor),3,0,Math.PI*2);ctx.fill();
      }
      ctx.lineWidth=1;ctx.strokeStyle='#e0ebf066';ctx.beginPath();ctx.moveTo(x(index),top);ctx.lineTo(x(index),h-bottom);ctx.stroke();
      ctx.fillStyle=C.muted;ctx.textAlign='center';for(let i=0;i<clip.frames.length;i+=Math.max(1,Math.ceil(clip.frames.length/5)))ctx.fillText(clip.frames[i].time_s.toFixed(1)+'s',x(i),h-9);
      $('causalTrend').setAttribute('aria-label',`截至${clip.frames[index].time_s.toFixed(1)}秒，最大分数${samples.at(-1).score.toFixed(3)}，有效区占比${samples.at(-1).valid}/64，相交支持区${samples.at(-1).possible}/64`);
    }
  }
  root.Telemetry={Instruments,summary,history,transitions,supportVertices,volumeSVG};
})(typeof window==='undefined'?{}:window);
