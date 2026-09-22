/* Observable-only working-principle graphics. Sealed decisions are never recomputed. */
(function(root){
  'use strict';
  const n=(v,p=3)=>Number.isFinite(v)?v.toFixed(p):'—';
  const escape=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function model(frame,zone,threshold){
    const a=frame.zones[zone],d=frame.decision;
    let winner=null;
    frame.zones.forEach((z,i)=>{if(z.possible&&(winner===null||z.joint>frame.zones[winner].joint))winner=i;});
    return {zone,winner,value:frame.values[zone],interval:a.interval,depth:a.depth,angular:a.angular_given_depth,joint:a.joint,
      score:frame.score,threshold,definite:frame.raw.definite_zones,possible:frame.raw.alert,
      strong:d.strong,previousStrong:d.previous_strong,previousId:d.previous_frame_id,held:d.held_only,alert:d.alert,unknown:d.unknown,
      status:!a.interval?'无有效返回':a.depth===0?'有效返回 · 无深度交集':'有效返回 · 有深度交集'};
  }
  function projection(zone,geometry,volume,side){
    const W=320,H=248,pad={l:36,r:14,t:36,b:30},far=Math.max(4.2,Math.ceil(zone.interval?.[1]||0));
    const slopes=side?geometry.slopes_b:geometry.slopes_a,axis=side?volume.y:volume.x;
    const extent=Math.max(1.25,Math.abs(axis[0]),Math.abs(axis[1]),...(zone.interval?slopes.map(s=>Math.abs(s*zone.interval[1])):[0]));
    const cross=[-extent,extent],pw=W-pad.l-pad.r,ph=H-pad.t-pad.b;
    const xy=(v,z)=>side?[pad.l+z/far*pw,pad.t+(v-cross[0])/(cross[1]-cross[0])*ph]:[pad.l+(v-cross[0])/(cross[1]-cross[0])*pw,pad.t+(1-z/far)*ph];
    const points=a=>a.map(([v,z])=>xy(v,z).map(x=>x.toFixed(2)).join(',')).join(' ');
    const box=[[axis[0],volume.z[0]],[axis[1],volume.z[0]],[axis[1],volume.z[1]],[axis[0],volume.z[1]]];
    const id='mechanism-'+(side?'side':'top');
    let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="选中区${side?'侧视 Y/Z':'俯视 X/Z'}投影"><defs><clipPath id="${id}"><polygon points="${points(box)}"/></clipPath></defs><text x="${pad.l}" y="20" class="projection-title">${side?'侧视 / Y–Z':'俯视 / X–Z'}</text>`;
    for(let z=0;z<=far;z+=far>6?2:1){const p=xy(cross[0],z),q=xy(cross[1],z);svg+=`<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="#344940"/>`;
      svg+=side?`<text x="${p[0]}" y="${H-11}" text-anchor="middle">${z}m</text>`:`<text x="${pad.l-8}" y="${p[1]+4}" text-anchor="end">${z}m</text>`;
    }
    svg+=`<polygon points="${points(box)}" fill="#76cdb81c" stroke="#76cdb8" stroke-width="1.5"/>`;
    if(zone.interval){const [lo,hi]=zone.interval,[a,b]=slopes,polygon=points([[a*lo,lo],[b*lo,lo],[b*hi,hi],[a*hi,hi]]);
      const camera=xy(0,0);for(const s of slopes){const p=xy(s*hi,hi);svg+=`<path d="M${camera.join(',')}L${p.join(',')}" stroke="#e3bb6c66" stroke-dasharray="4 5"/>`;}
      svg+=`<polygon points="${polygon}" fill="#e3bb6c22" stroke="#e3bb6c" stroke-width="1.5"/><polygon points="${polygon}" clip-path="url(#${id})" fill="#e3bb6c55"/>`;
    }else svg+=`<text x="${W/2}" y="${H/2}" text-anchor="middle">无有效返回</text>`;
    const c=xy(0,0);svg+=`<circle cx="${c[0]}" cy="${c[1]}" r="3" fill="#fff"/><text x="${W-18}" y="20" text-anchor="end">${side?'Y↓  Z→':'X→  Z↑'}</text></svg>`;
    return svg;
  }
  function history(clip,index){return clip.frames.slice(0,index+1).map((f,i)=>({index:i,time:f.time_s,strong:f.decision.strong,held:f.decision.held_only,alert:f.decision.alert}));}
  function render({frame,clip,index,zone,geometry,threshold,title,onZone,onSeek}){
    const $=id=>document.getElementById(id),m=model(frame,zone,threshold);
    $('mechanismScene').textContent=title;$('mechanismFrame').textContent=`${frame.id} · ${n(frame.time_s,1)} s · ${index+1} / ${clip.frames.length}`;
    $('mechanismSeek').max=clip.frames.length-1;$('mechanismSeek').value=index;$('mechanismPrev').disabled=index===0;$('mechanismNext').disabled=index===clip.frames.length-1;
    $('mechanismGrid').innerHTML=frame.values.map((v,i)=>`<button class="${i===zone?'selected ':''}${v===null?'missing ':''}${i===m.winner?'winner':''}" aria-pressed="${i===zone}" aria-label="原理区域 ${i}，${v===null?'无有效返回':n(v,2)+'米'}" data-mechanism-zone="${i}" style="--level:${v===null?0:Math.max(0,1-Math.min(v,9)/9)}">${v===null?'—':n(v,1)}</button>`).join('');
    $('mechanismGrid').querySelectorAll('button').forEach(b=>b.onclick=()=>onZone(Number(b.dataset.mechanismZone)));
    $('mechanismRange').innerHTML=`<span>选中 z${zone} · ${m.status}</span><strong>${n(m.value)} <small>m</small></strong><p>完整区间 <b>${m.interval?m.interval.map(v=>n(v)).join(' – ')+' m':'—'}</b></p>`;
    $('mechanismWinner').disabled=m.winner===null;$('mechanismWinner').textContent=m.winner===null?'无可能相交区':`选择最大分数区 z${m.winner}`;$('mechanismWinner').onclick=()=>{if(m.winner!==null)onZone(m.winner);};
    $('mechanismProjection').innerHTML=projection(frame.zones[zone],geometry.zone_geometry[zone],geometry.volume,false)+projection(frame.zones[zone],geometry.zone_geometry[zone],geometry.volume,true);
    $('mechanismFactors').innerHTML=`<div><span>深度占比</span><strong>${n(m.depth)}</strong><i style="--value:${m.depth||0}"></i></div><b>×</b><div><span>平均角域重叠</span><strong>${n(m.angular)}</strong><i style="--value:${m.angular||0}"></i></div><b>=</b><div><span>选中区分数</span><strong>${n(m.joint)}</strong><i style="--value:${m.joint||0}"></i></div>`;
    const scoreRoute=!!m.possible&&m.score>=threshold;
    $('mechanismScore').innerHTML=`<span>全帧最大分数${m.winner===null?'':` · z${m.winner}`}</span><strong>${n(m.score,6)}</strong><div class="mechanism-meter"><i style="width:${Math.min(1,Math.max(0,m.score))*100}%"></i><b style="left:${threshold*100}%"></b></div><small>0 <span>阈值 ${n(threshold,6)}</span> 1</small>`;
    $('mechanismRoutes').innerHTML=`<div class="${m.definite?'active':''}"><b>${m.definite?'✓':'—'}</b><span>确定支持<small>${m.definite} 区 · 直接触发</small></span></div><i>或</i><div class="${scoreRoute?'active':''}"><b>${scoreRoute?'✓':'—'}</b><span>分数通路<small>可能相交 + 达到阈值</small></span></div>`;
    $('mechanismStrong').innerHTML=`<span>当前真实强证据</span><strong>${m.strong?'有':'无'}</strong>`;$('mechanismStrong').dataset.active=String(m.strong);
    const status=m.held?'一帧保持':m.alert?'当前证据提醒':'未触发提醒';
    $('mechanismMemory').innerHTML=`<div class="memory-node ${m.previousStrong?'on':''}"><span>上一相邻采样</span><strong>${m.previousStrong?'有真实强证据':'无可用强证据'}</strong><small>${m.previousStrong?escape(m.previousId)+' · 允许续 0.2 s':'不从上次告警继承状态'}</small></div><span class="circuit-operator">或</span><div class="memory-node ${m.strong?'on':''}"><span>当前采样</span><strong>${m.strong?'有真实强证据':'无真实强证据'}</strong><small>${escape(frame.id)} · ${n(frame.time_s,1)} s</small></div><span class="circuit-operator">→</span><div class="memory-node result ${m.held?'held':m.alert?'on':''}"><span>最终告警</span><strong>${status}</strong><small>${m.unknown?'空间 UNKNOWN 保留':'当前有确定通道支持'}</small></div>`;
    const past=history(clip,index),rows=[['strong','真实强证据'],['held','仅一帧保持'],['alert','最终告警']];
    $('mechanismHistory').style.setProperty('--samples',clip.frames.length);
    $('mechanismHistory').innerHTML=rows.map(([key,label])=>`<div class="history-lane"><span>${label}</span>${Array.from({length:clip.frames.length},(_,i)=>{const s=past[i];return `<button ${s?'':'disabled'} class="${s?.[key]?'lit '+key:''} ${i===index?'current':''}" data-mechanism-time="${i}" aria-label="${label}，采样 ${i+1}${s?'，'+(s[key]?'有':'无'):'，未来采样'}"></button>`;}).join('')}</div>`).join('')+`<div class="history-times"><span>时间 / s</span>${Array.from({length:clip.frames.length},(_,i)=>`<span>${past[i]&&(i%4===0||i===index)?n(past[i].time,1):''}</span>`).join('')}</div><p>点击当前及之前采样定位 · 当前 ${n(frame.time_s,1)} s <span>空白未来不参与当前读出</span></p>`;
    $('mechanismHistory').querySelectorAll('button:not(:disabled)').forEach(b=>b.onclick=()=>onSeek(Number(b.dataset.mechanismTime)));
  }
  const api={model,projection,history,render};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  else root.Mechanism=api;
})(typeof window==='undefined'?{}:window);
