(function (root) {
  'use strict';
  const names = {head_horizontal:'横向悬空体', head_hanging_plane:'悬挂面', body_protruding_plane:'身体突出面', body_suspended_solid:'身体悬空体'};
  const tags = {rescue:'新增正确报警', extra_fp:'新增误报', miss:'整段漏检', new_event:'补回事件'};
  const sec = v => v == null ? '—' : `${Number(v).toFixed(1)} 秒`;
  const eventValue = (m, key) => !m.eventCount ? '无事件' : m[key] == null ? '未检出' : sec(m[key]);
  function matches(c, f) {
    const haystack = [c.id,c.layoutId,c.typeId,c.layer,c.relation,names[c.typeId] || ''].join(' ').toLowerCase();
    return (f.group === 'all' || c.stratum === f.group) && (f.layer === 'all' || c.layer === f.layer) &&
      (f.outcome === 'all' || c.tags.includes(f.outcome)) && haystack.includes(f.search.trim().toLowerCase());
  }
  function outcome(f) {
    if (!f.aHold && f.andHold) return f.truth ? ['rescue','新增正确报警'] : ['extra_fp','新增误报'];
    if (f.truth && !f.andHold) return ['miss','仍漏报'];
    if (!f.truth && f.aHold) return ['extra_fp','基线已有误报'];
    return f.truth ? ['', '基线已检出'] : ['', '当前未报警'];
  }
  function reason(f) {
    if (f.andCurrent) return f.a ? 'A 在当前帧已触发；对照保留这一报警。' : 'B 与 N 在当前帧同时同意，触发补报。';
    return f.andHold ? '来自上一帧当前判断的单帧保持；保持结果不会再次延长。' : '当前判断及上一帧当前判断均未触发。';
  }
  const core = {matches,outcome,reason,eventValue,sec};
  if (typeof module !== 'undefined' && module.exports) module.exports = core;
  if (!root || !root.document) return;
  const $ = id => root.document.getElementById(id);
  const escape = v => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const data = root.REPLAY_DATA;
  if (!data || !data.clips || data.clips.length !== 48 || data.clips.reduce((n,c) => n+c.frames.length,0) !== 1152) {
    $('fatal').hidden=false; $('fatal').textContent='数据不完整。请完整解压后打开 index.html。'; return;
  }
  const state = {clip:data.clips[0],frame:0,mode:'A',timer:null};
  function filters() { return {group:$('group-filter').value,layer:$('layer-filter').value,outcome:$('outcome-filter').value,search:$('search').value}; }
  function stop() { if(state.timer !== null) root.clearInterval(state.timer); state.timer=null; $('play').textContent='▶ 播放'; $('play').setAttribute('aria-label','播放'); }
  function seek(i) { stop(); state.frame=Math.max(0,Math.min(state.clip.frames.length-1,i)); renderFrame(); }
  function select(c) { stop(); state.clip=c;state.frame=0;renderList();renderClip(); }
  function renderList() {
    const visible=data.clips.filter(c=>matches(c,filters()));
    $('filter-count').textContent=`${visible.length} / ${data.clips.length}`;
    $('clip-list').replaceChildren();
    visible.forEach(c=>{
      const b=root.document.createElement('button');b.type='button';b.className='clip-item';b.setAttribute('aria-current',String(c===state.clip));
      b.innerHTML=`<strong>${escape(names[c.typeId] || c.typeId)}</strong><span class="clip-sub">${escape(c.layer)} · ${escape(c.stratum)} · ${escape(c.relation)}</span><span class="clip-sub">${escape(c.layoutId)}</span><span class="mini-tags">${c.tags.filter(t=>tags[t]).map(t=>`<span class="mini-tag ${t==='extra_fp'||t==='miss'?'bad':'good'}">${tags[t]}</span>`).join('')}</span>`;
      b.addEventListener('click',()=>select(c));$('clip-list').append(b);
    });
    if(!visible.length) $('clip-list').textContent='没有匹配的片段。当前画面保留原片段，可重置筛选。';
    return visible;
  }
  function applyFilter(jump) {
    const visible=renderList();
    if(visible.length && (!visible.includes(state.clip) || jump)) select(visible[0]);
    if(jump && visible.length) {
      const tag=$('outcome-filter').value;
      const i=state.clip.frames.findIndex(f=>tag==='rescue'?f.truth&&!f.aHold&&f.andHold:tag==='extra_fp'?!f.truth&&!f.aHold&&f.andHold:f.truth);
      if(i>=0) seek(i);
    }
  }
  function renderClip() {
    const c=state.clip;
    $('clip-context').textContent=`${c.stratum} / ${c.layer} / ${c.relation}`;
    $('clip-title').textContent=names[c.typeId] || c.typeId;$('clip-id').textContent=c.id;
    $('frame-slider').max=c.frames.length-1;
    $('clip-metrics').innerHTML=['A','AND'].map(k=>{
      const m=c.metrics[k];return `<tr><th>${k==='A'?'A 保留基线':'一致性补报'}</th><td>${m.TP} / ${m.FP} / ${m.FN}</td><td>${eventValue(m,'firstEventAlertTime')}</td><td>${eventValue(m,'delay')}</td><td>${m.fpSegments}</td><td>${sec(m.fpSeconds)}</td></tr>`;
    }).join('');
    $('event-note').textContent=c.metrics.A.eventCount ? `首次相交标注：${sec(c.metrics.A.eventEntry)}。事件前的误报不算检出；延迟以采样时刻计算。` : '此片段没有相交事件。未报警不构成安全判断。';
    $('fp-details').innerHTML=['A','AND'].map(k=>`<p>${k==='A'?'A':'对照'} 误报区间：${c.metrics[k].fpIntervals.length?c.metrics[k].fpIntervals.map(s=>`[${Number(s.startTime).toFixed(1)}, ${Number(s.endExclusive).toFixed(1)}) 秒`).join('；'):'无'}</p>`).join('');
    $('timeline').replaceChildren();
    c.frames.forEach((f,i)=>{
      const b=root.document.createElement('button');b.type='button';b.className='time-cell';
      b.title=`${sec(f.time)} · 相交 ${+f.truth} · A ${+f.aHold} · 对照 ${+f.andHold}`;b.setAttribute('aria-label',b.title);
      const flagClass=v=>v?(f.truth?'tp':'fp'):(f.truth?'fn':'');
      b.innerHTML=`<span class="${f.truth?'truth':''}"></span><span class="${flagClass(f.aHold)}"></span><span class="${flagClass(f.andHold)}"></span>`;
      b.addEventListener('click',()=>seek(i));$('timeline').append(b);
    });renderFrame();
  }
  function renderFrame() {
    const f=state.clip.frames[state.frame];const selected=state.mode==='A'?f.aHold:f.andHold;
    $('frame-image').src=f.image;$('frame-image').alt=`${state.clip.id}，${sec(f.time)}，保存的合成 RGB`;
    $('frame-id').textContent=`${f.id} · ${state.frame+1} / ${state.clip.frames.length}`;
    $('active-status').textContent=`${state.mode==='A'?'A':'对照'} · ${selected?'报警':'未报警'}`;$('active-status').className=`active-status${selected?' alert':''}`;
    $('truth-status').textContent=`评估标注：${f.truth?'与通道相交':'未相交'}`;
    $('time-readout').textContent=`${f.time.toFixed(1)} / ${state.clip.frames.at(-1).time.toFixed(1)} 秒`;
    $('frame-slider').value=state.frame;$('step-back').disabled=state.frame===0;$('step-forward').disabled=state.frame===state.clip.frames.length-1;
    [...$('timeline').children].forEach((el,i)=>el.setAttribute('aria-current',String(i===state.frame)));
    const [cls,label]=outcome(f);$('outcome').className=`outcome ${cls}`;$('outcome').textContent=label;
    $('frame-alerts').innerHTML=[['A',f.aHold],['AND',f.andHold]].map(([k,v])=>`<div class="alert-line${v?' on':''}${state.mode===k?' selected':''}"><span>${k==='A'?'A 保留基线':'一致性补报'}</span><strong>${v?'报警':'未报警'}</strong></div>`).join('');
    $('current-bits').innerHTML=[['A',f.a],['B',f.b],['N',f.n]].map(([k,v])=>`<span class="bit${v?' on':''}">${k} ${v?'1':'0'}</span>`).join('');
    $('reason').textContent=reason(f);$('unknown-status').textContent=f.unknown?'UNKNOWN · 当前证据不明确':'当前证据可判定';
    $('mode-a').setAttribute('aria-pressed',String(state.mode==='A'));$('mode-and').setAttribute('aria-pressed',String(state.mode==='AND'));
  }
  function play() {
    if(state.timer!==null){stop();return;}
    if(state.frame===state.clip.frames.length-1) state.frame=0;
    $('play').textContent='Ⅱ 暂停';$('play').setAttribute('aria-label','暂停');renderFrame();
    state.timer=root.setInterval(()=>{state.frame++;renderFrame();if(state.frame===state.clip.frames.length-1)stop();},data.meta.dt*1000/Number($('speed').value));
  }
  $('frame-image').addEventListener('error',()=>{$('image-error').hidden=false;});
  $('frame-image').addEventListener('load',()=>{$('image-error').hidden=true;});
  $('play').addEventListener('click',play);$('step-back').addEventListener('click',()=>seek(state.frame-1));$('step-forward').addEventListener('click',()=>seek(state.frame+1));
  $('frame-slider').addEventListener('input',e=>seek(Number(e.target.value)));
  $('speed').addEventListener('change',()=>{if(state.timer!==null){stop();play();}});
  $('mode-a').addEventListener('click',()=>{state.mode='A';renderFrame();});$('mode-and').addEventListener('click',()=>{state.mode='AND';renderFrame();});
  ['group-filter','layer-filter','outcome-filter'].forEach(id=>$(id).addEventListener('change',()=>applyFilter(false)));
  $('search').addEventListener('input',()=>applyFilter(false));
  root.document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{
    $('group-filter').value='all';$('layer-filter').value='all';$('search').value='';$('outcome-filter').value=b.dataset.filter;applyFilter(true);
  }));
  $('clear-filters').addEventListener('click',()=>{['group-filter','layer-filter','outcome-filter'].forEach(id=>$(id).value='all');$('search').value='';applyFilter(false);});
  root.document.addEventListener('keydown',e=>{if(/INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName))return;if(e.key==='ArrowLeft'){e.preventDefault();seek(state.frame-1);}if(e.key==='ArrowRight'){e.preventDefault();seek(state.frame+1);}});
  root.addEventListener('pagehide',stop);
  $('dataset-count').textContent=`${data.meta.layouts} 布局 · ${data.meta.clips} 片段 · ${data.meta.frames} 帧`;
  $('scorecards').innerHTML=['Core','Boundary'].map(g=>{
    const s=data.summary[g];return `<article class="scorecard"><div class="score-title"><strong>${g}</strong><span>全量 · 保持后</span></div><div class="score-compare">${['A','AND'].map((k,i)=>`${i?'<span class="score-arrow">→</span>':''}<div class="score-arm"><small>${k==='A'?'A 保留基线':'一致性补报'}</small><strong>${s[k].TP} / ${s[k].FP} / ${s[k].FN}</strong><p>正确 / 误报 / 漏报</p><p>事件 ${s[k].detectedEvents} / ${s[k].eventCount} · 误报 ${s[k].fpSegments} 段 / ${sec(s[k].fpSeconds)}</p></div>`).join('')}</div><p class="score-result${g==='Boundary'?' fail':''}">${g==='Core'?'局部补漏成立；新增成本约束通过':'新增误报成本超预算；未整体晋级'}</p></article>`;
  }).join('');
  $('budget-note').textContent=`完整数据结论固定显示：Core 有局部收益，Boundary 的新增误报及片段超预算。对照尚未替换主线 A。`;
  $('source-description').textContent=data.meta.source+' · '+data.meta.limits;
  renderList();renderClip();
})(typeof window !== 'undefined' ? window : null);
