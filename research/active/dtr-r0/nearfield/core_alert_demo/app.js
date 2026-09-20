/* Offline replay of sealed decisions. UI never derives alerts from truth or rounded scores. */
(function () {
  'use strict';
  const COLORS={teal:'#4ed4bc',amber:'#f3bd64',red:'#f27878',pink:'#ca9be8',muted:'#8b9bb0',blue:'#81a9ed'};
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fixed=(v,n=3)=>Number.isFinite(v)?v.toFixed(n):'—';
  const pct=(v,n=1)=>Number.isFinite(v)?(v*100).toFixed(n)+'%':'—';
  const stateFor=f=>f.decision.held_only?'held':f.decision.strong?'strong':'silent';
  const hasTruth=f=>typeof f.evaluation.truth==='boolean';
  const outcomeFor=(truth,alert)=>typeof truth!=='boolean'?'UNLABELED':truth?(alert?'TP':'FN'):(alert?'FP':'TN');
  const evaluatedCohorts=cohorts=>cohorts.filter(c=>!c.illustrative&&c.metrics);
  const flagFor=(f,arm)=>f.decision[arm==='hold'?'alert':arm];
  function clipStats(clip,arm='hold') {
    let tp=0,fp=0,fn=0,segments=0,lastFP=false;
    for(const f of clip.frames){if(!hasTruth(f)){lastFP=false;continue;}const a=flagFor(f,arm),t=f.evaluation.truth;tp+=+(a&&t);fp+=+(a&&!t);fn+=+(!a&&t);segments+=+(a&&!t&&!lastFP);lastFP=a&&!t;}
    return {tp,fp,fn,segments};
  }
  const dominantZone=f=>{
    let best=-1;
    for(let z=0;z<64;z++) if(f.zones[z].possible&&(best<0||f.zones[z].joint>f.zones[best].joint))best=z;
    if(best<0)best=f.values[35]!==null?35:f.values.findIndex(v=>v!==null);
    return best<0?35:best;
  };
  if(typeof module!=='undefined'&&module.exports)module.exports={stateFor,outcomeFor,flagFor,clipStats,dominantZone,hasTruth,evaluatedCohorts};
  if(typeof document==='undefined')return;
  const $=id=>document.getElementById(id),sourceData=window.CORE_DEMO_DATA;
  const extra=window.CORE_DEMO_EXTRA;
  const data=sourceData?{...sourceData,cohorts:[...sourceData.cohorts,...(extra&&extra.clips&&extra.clips.length?[extra]:[])]}:null;
  if(!data){document.querySelector('main').innerHTML='<div class="panel prose"><h1>回放数据尚未就绪</h1><p>请保留整个演示文件夹，确认 demo-data.js 与 index.html 位于同一目录。</p></div>';return;}
  $('collectionCount').textContent=String(data.cohorts.length).padStart(2,'0');
  const S={cohort:0,clip:0,frame:0,zone:35,autoZone:true,layout:'ALL',family:'ALL',page:'gallery',playing:false,timer:null,tour:null};
  const currentCohort=()=>data.cohorts[S.cohort],currentClip=()=>currentCohort().clips[S.clip],currentFrame=()=>currentClip().frames[S.frame];
  const tone=new EventNotifications.Tone({makeContext:()=>new (window.AudioContext||window.webkitAudioContext)(),
    onStatus:text=>{$('notificationStatus').textContent=text;}});
  const notifications=new EventNotifications.Scheduler({emit:event=>tone.notify(event,notifications),cancel:()=>tone.cancel()});
  function notifyFrame(){
    const f=currentFrame();notifications.step({clipId:currentCohort().id+'/'+currentClip().id,
      frameIndex:S.frame,timeS:f.time_s,alert:f.decision.alert});
    $('notificationStatus').textContent=tone.enabled?(f.decision.alert?'连续告警 · 本段不重复提示':'等待下一段告警'):'提示音已关闭';
  }
  $('notificationSound').onchange=async e=>{
    const enabled=e.target.checked;
    if(enabled){if(!await tone.enable())$('notificationSound').checked=false;}
    else{notifications.silence();tone.disable();}
    $('notificationStatus').textContent=tone.enabled?'已开启 · 下一段告警提示一次':'提示音已关闭';
  };
  window.addEventListener('pagehide',()=>{notifications.reset();tone.close();});
  const instruments=new window.Telemetry.Instruments({geometry:data.geometry,threshold:data.thresholds.strong,
    onZone:zone=>{S.zone=zone;S.autoZone=false;renderZone();renderPrinciple();},onSeek:frame=>setFrame(frame)});
  function renderInstruments(){instruments.render(currentClip(),S.frame,S.zone);instruments.status(S.playing);}
  const typeNames={head_horizontal:'横向悬空体',head_hanging_plane:'悬挂面',head_protruding_edge:'头部突出边缘',body_protruding_plane:'身体突出面',body_suspended_solid:'身体悬空体',body_large_solid:'身体大型实体'};
  const layoutNames={INSIDE:'主体 · 相交布局',OUTSIDE:'主体 · 通道外布局',BOUNDARY:'挑战 · 边界接触'};
  const clipTitle=c=>c.title||(c.description?(c.label||c.type):(typeNames[c.type]||c.label||c.type)+' · '+({INSIDE:'走近障碍',OUTSIDE:'从旁经过',BOUNDARY:'接近通道边缘'}[c.layout]||c.layout));
  const clipDescription=c=>c.description||({INSIDE:'沿相机前向接近，观察测距支持进入通道后，强证据与提醒如何变化。',OUTSIDE:'障碍保持在通道侧方；对照区域支持与分数，观察是否仍会提醒。',BOUNDARY:'支撑擦过通道边缘，观察分数、提醒与评估边界之间的区别。'}[c.layout]||'播放完整原始片段，查看观测与固定策略的响应。');
  const thumbnail=c=>c.frames[c.preview_index??Math.floor(c.frames.length*.55)].rgb;
  const visibleClips=()=>currentCohort().clips.map((c,i)=>({c,i})).filter(({c})=>(S.layout==='ALL'||c.layout===S.layout)&&(S.family==='ALL'||c.layer===S.family));
  function renderGallery(){const co=currentCohort(),clips=visibleClips();$('galleryCohort').textContent=co.title;$('galleryDescription').textContent=co.illustrative?'2 张地图 · 6 处位置 · 1080p RGB · 未标注，不计入验证结果。':co.subtitle+' · 六种障碍形态，每个完整片段均可打开。';$('galleryCount').textContent=clips.length+' 个完整片段';$('galleryGrid').innerHTML=clips.map(({c,i})=>`<button class="gallery-card" data-gallery-clip="${i}"><div class="gallery-image"><img src="${esc(thumbnail(c))}" alt="${esc(clipTitle(c))}" loading="lazy"><span>${co.illustrative?'机制演示 · 未标注':esc(c.layer+' / '+c.layout)}</span><b>↗</b></div><div class="gallery-card-copy"><small>${String(i+1).padStart(2,'0')} · ${c.frames.length} 个采样</small><h2>${esc(clipTitle(c))}</h2><p>${esc(clipDescription(c))}</p>${c.motion?`<div class="motion-caption">运动 · ${esc(c.motion)}</div>`:''}<span>完整回放 <i>→</i></span></div></button>`).join('')||'<p class="empty">当前筛选没有场景，请选择全部类型。</p>';$('galleryGrid').querySelectorAll('[data-gallery-clip]').forEach(b=>b.onclick=()=>{selectClip(Number(b.dataset.galleryClip));setPage('replay');});}
  function renderTour(){const active=!!S.tour;$('tourStatus').hidden=!active;if(active)$('tourLabel').textContent=`连续导览 ${S.tour.index+1} / ${S.tour.clips.length} · 每个片段独立重置历史`;}
  async function startTour(){const clips=visibleClips().map(({i})=>i);if(!clips.length)return;setPage('replay');if(!await selectClip(clips[0]))return;S.tour={clips,index:0};$('loop').checked=false;renderTour();preload();play();}
  let toastTimer;
  function toast(text){$('toast').textContent=text;$('toast').classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('show'),2600);}
  const images=new ReplayImages.DecodedFrames();
  const presenter=new ReplayImages.FramePresenter(images,{
    busy:visible=>{const box=$('imageLoading');box.style.display=visible?'grid':'none';box.textContent='正在准备高清画面…';},
    failure:()=>{stop();$('imageLoading').style.display='grid';$('imageLoading').textContent='场景图像不可用；请确认 assets 文件夹完整。';}
  });
  function setNavigation(pending){for(const id of ['play','prevFrame','nextFrame','seek'])$(id).disabled=pending;}
  function navigationReady(promise){const token=presenter.token;setNavigation(true);return promise.finally(()=>{if(token===presenter.token)setNavigation(false);});}
  function stop(){notifications.silence();$('notificationStatus').textContent=tone.enabled?'已暂停 · 提示已取消':'提示音已关闭';setNavigation(false);$('cohort').value=String(S.cohort);S.playing=false;clearTimeout(S.timer);S.timer=null;presenter.cancel();$('play').textContent='▶ 播放';$('play').setAttribute('aria-label','播放');instruments.status(false);}
  function playingLabel(){$('play').textContent='Ⅱ 暂停';$('play').setAttribute('aria-label','暂停');instruments.status(true);}
  async function play(){
    if(S.playing){stop();return;}
    S.playing=true;playingLabel();
    if(S.frame===currentClip().frames.length-1&&!await setFrame(0,false))return;
    if(S.playing){notifyFrame();schedule();}
  }
  function schedule(){
    clearTimeout(S.timer);if(!S.playing)return;
    const interval=((currentClip().frames[S.frame+1]?.time_s-currentFrame().time_s)||.2)*1000/Number($('speed').value);
    S.timer=setTimeout(async()=>{
      if(!S.playing)return;
      const end=currentClip().frames.length-1;
      if(S.frame>=end){
        if(S.tour){
          const tour=S.tour;
          if(tour.index+1>=tour.clips.length){stop();S.tour=null;renderTour();toast('已到连续导览终点。');return;}
          tour.index++;
          if(!await selectClip(tour.clips[tour.index]))return;
          S.tour=tour;renderTour();preload();S.playing=true;playingLabel();notifyFrame();
        }else if($('loop').checked){if(!await setFrame(0,false))return;}
        else{stop();return;}
      }else if(!await setFrame(S.frame+1,false))return;
      if(S.playing)schedule();
    },Math.max(1,interval));
  }
  function setFrame(n,manual=true){
    if(manual)stop();
    const frame=Math.max(0,Math.min(currentClip().frames.length-1,Math.round(n)));
    if(manual||frame<=S.frame)notifications.reset();
    return presenter.show(currentClip().frames[frame].rgb,image=>{S.frame=frame;renderFrame(image);if(S.playing)notifyFrame();});
  }
  function setPage(page){if(page!=='replay'){stop();S.tour=null;renderTour();}S.page=page;document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id==='page-'+page));document.querySelectorAll('.nav').forEach(b=>b.classList.toggle('active',b.dataset.page===page));if(page==='gallery')renderGallery();if(page==='results')renderResults();if(page==='principle')renderPrinciple();if(page==='replay')renderInstruments();window.scrollTo(0,0);}
  function selectClip(index,frame=0){
    stop();notifications.reset();S.tour=null;renderTour();
    const clip=currentCohort().clips[index];frame=Math.max(0,Math.min(clip.frames.length-1,frame));
    const ready=presenter.show(clip.frames[frame].rgb,image=>{S.autoZone=true;S.clip=index;S.frame=frame;renderSceneList();renderFrame(image);});
    warmClip(clip,frame,currentCohort().clips[index+1]);return navigationReady(ready);
  }
  function selectCohort(index,clip=0,frame=0){
    stop();notifications.reset();S.tour=null;renderTour();const co=data.cohorts[index],selected=co.clips[clip];
    const ready=presenter.show(selected.frames[frame].rgb,image=>{
      S.autoZone=true;S.cohort=index;S.clip=clip;S.frame=frame;S.layout='ALL';S.family='ALL';
      $('cohort').value=String(index);$('familyFilter').value='ALL';document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b.dataset.filter==='ALL'));
      $('cohortNote').textContent=co.subtitle;renderKPIs();renderSceneList();renderGallery();renderFrame(image);if(S.page==='results')renderResults();
    });
    warmClip(selected,frame,co.clips[clip+1]);return navigationReady(ready);
  }
  function warmClip(clip,frame,next){images.warm([...clip.frames.slice(frame),...clip.frames.slice(0,frame),...(next?.frames.slice(0,4)||[])].map(f=>f.rgb));}
  function preload(){const next=S.tour?S.tour.clips[S.tour.index+1]:S.clip+1;warmClip(currentClip(),S.frame,currentCohort().clips[next]);}
  function renderKPIs(){if(currentCohort().illustrative||!currentCohort().metrics){$('kpis').innerHTML='<div class="illustrative-banner"><span class="tag amber">动态机制演示 · 无评估真值</span><p>展示原生渲染场景、实际区域观测与同一冻结策略的响应。没有 TP / FP 或事件效果评价，原验证统计保持独立。</p></div>';return;}const m=currentCohort().metrics.core.hold,b=currentCohort().metrics.core.calibration;const reduction=b.FP?(b.FP-m.FP)/b.FP:null;
    $('kpis').innerHTML=`<div class="kpi"><span class="field-label">Core 全量 · ${m.frames} 帧 · 主体事件</span><strong>${m.events_detected}<em>/ ${m.event_count}</em></strong><small>${m.TP} TP / ${m.FN} FN · 全部主体布局</small></div><div class="kpi"><span class="field-label">候选误报帧</span><strong>${m.FP}<em>帧</em></strong><span class="change">↓ ${pct(reduction)}</span><small>Calibration ${b.FP} 帧 → 候选 ${m.FP} 帧</small></div><div class="kpi"><span class="field-label">主体正帧覆盖</span><strong>${pct(m.recall)}</strong><small>精确率 ${pct(m.precision)} · F1 ${pct(m.f1)}</small></div><div class="kpi"><span class="field-label">误报段 / 采样时长</span><strong>${m.false_segments}<em>段 / ${fixed(m.false_sampled_s,1)} s</em></strong><small>Calibration ${b.false_segments} 段 / ${fixed(b.false_sampled_s,1)} s</small></div>`;}
  function renderSceneList(){const co=currentCohort(),clips=visibleClips();$('sceneCount').textContent=`${clips.length} / ${co.clips.length}`;
    $('sceneList').innerHTML=clips.length?clips.map(({c,i})=>{const q=clipStats(c),evaluated=!co.illustrative&&c.frames.some(hasTruth);return `<button class="scene-item ${S.clip===i?'active':''}" data-clip="${i}" aria-current="${S.clip===i}"><img class="scene-thumb" src="${esc(thumbnail(c))}" alt="" loading="lazy"><div><strong>${esc(clipTitle(c))}</strong><span>${esc(c.layer)} · ${c.frames.length} 采样</span><br><span>${evaluated?`正例 ${q.tp}/${q.tp+q.fn} · 误报 ${q.fp} 帧`:'未标注 · 仅展示机制响应'}</span></div></button>`;}).join(''):'<p class="empty">当前筛选没有场景。</p>';
    $('sceneList').querySelectorAll('[data-clip]').forEach(b=>b.onclick=()=>{selectClip(Number(b.dataset.clip));setPage('replay');});}
  function loadImage(frame,image){
    $('fullImage').href=frame.rgb;$('fullImage').textContent=frame.rgb_size?.[0]>=1920?'1080p 原图 ↗':'640×360 原图 ↗';
    const old=$('sceneImage');
    image.alt=`${currentClip().label}，${frame.id}，${fixed(frame.time_s,1)}秒`;image.width=640;image.height=360;
    if(old!==image){old.removeAttribute('id');image.id='sceneImage';old.replaceWith(image);}
    $('imageLoading').style.display='none';
  }
  function renderFrame(image){$('guideNote').textContent='导览不改变统计范围；所有完整片段均可查看。';const f=currentFrame(),clip=currentClip();if(S.autoZone)S.zone=dominantZone(f);$('seek').max=clip.frames.length-1;$('seek').value=S.frame;$('timeLabel').textContent=`${fixed(f.time_s,1)} s / ${fixed(clip.frames.at(-1).time_s,1)} s · 第 ${S.frame+1} / ${clip.frames.length} 帧`;$('frameTag').textContent=f.id+' · '+fixed(f.time_s,1)+' s';$('sceneType').textContent=layoutNames[clip.layout]||'动态机制演示 · 未标注';$('sceneTitle').textContent=clipTitle(clip);$('sceneDescription').textContent=clipDescription(clip);$('sceneCaption').textContent=clip.environment||'相机前向 · 名义 45° ToF';$('motionLabel').textContent=clip.motion?'运动 · '+clip.motion:'按原始采样时间播放 · 不生成中间观测';
    const truth=f.evaluation.truth;const tb=$('truthBadge');tb.textContent=!hasTruth(f)?'无评估真值 · 不计成败':'评估真值：'+(truth?(f.evaluation.boundary?'接触 / 边界带':'已进入通道'):(clip.layout==='OUTSIDE'?'通道外':'入界前负期'));tb.className='tag '+(truth?'pink':'neutral');$('showTruth').disabled=!hasTruth(f);if(!hasTruth(f))$('showTruth').checked=false;loadImage(f,image);renderDecision();renderZone();renderTimeline();renderPrinciple();}
  function renderDecision(){const f=currentFrame(),d=f.decision,state=stateFor(f),definite=f.raw.definite_zones>0;const hero=$('decisionHero');hero.className='decision-hero '+state;
    $('decisionSymbol').textContent=state==='held'?'↳':state==='strong'?'!':'−';$('decisionKicker').textContent=state==='held'?'来自上一采样':state==='strong'?'来自当前采样':'当前采样';$('decisionTitle').textContent=state==='held'?'一帧保持提醒':state==='strong'?(definite?'确定支持提醒':'强分数提醒'):'当前不触发提醒';
    const previous=S.frame>0?currentClip().frames[S.frame-1]:null;
    $('decisionExplanation').textContent=state==='held'?`当前分数低于强证据线；上一采样 ${d.previous_frame_id} 提供真实强证据，允许延续 0.2 秒。保持本身不会续期。`:state==='strong'?(definite?'当前存在完整落在通道内的确定支持，直接提醒，不依赖分数阈值。':'当前存在可能相交的返回，且几何分数达到固定强证据阈值。'):(!f.raw.alert?'当前没有可能相交的返回，也没有可用的上一帧强证据。未提醒不等于通道畅通。':'当前几何分数未达强证据线，上一采样也没有可用于保持的强证据。');
    $('frameScore').textContent=fixed(f.score,6);$('scoreFill').style.width=Math.max(0,Math.min(1,f.score))*100+'%';$('scoreFill').style.background=state==='held'?COLORS.amber:COLORS.teal;
    const outcome=outcomeFor(f.evaluation.truth,d.alert);$('outcomeBadge').textContent=outcome==='UNLABELED'?'响应展示 · 未评价':'评估 '+outcome;$('outcomeBadge').className='tag '+(outcome==='FP'?'red':outcome==='FN'?'pink':outcome==='TN'||outcome==='UNLABELED'?'neutral':'');
    $('logicRow').innerHTML=`<div class="logic-cell ${d.strong?'on':''}">当前强证据<b>${d.strong?(definite?'有 · 确定支持':'有 · 分数达标'):'无'}</b></div><div class="logic-cell ${d.previous_strong?'on':''}">上一采样强证据<b>${d.previous_strong?'有 · '+esc(d.previous_frame_id):'无可用强证据'}</b></div>`;
    $('methodComparison').innerHTML=[['Calibration',d.calibration,false],['固定强阈值',d.strong,false],['强阈值 + 保持',d.alert,d.held_only]].map(([name,on,held])=>`<div>${name}<b class="${held?'held-text':on?'yes':'no'}">${held?'保持提醒':on?'提醒':'未提醒'}</b></div>`).join('');
    $('uncertainty').textContent=d.unknown?'UNKNOWN 保留 · 当前空间证据仍不确定，提醒不会把它变成已确认占据或畅通。':'当前存在确定通道支持 · 此状态仅针对本采样返回，不代表整条通道均已观测。';
  }
  function zoneColor(f,z){if(f.values[z]===null)return '#253244';if($('heatMode').value==='score'){const n=f.zones[z].joint;return `hsl(${170+18*(1-n)},${25+35*n}%,${13+32*n}%)`;}const t=Math.min(8,Math.max(0,f.values[z]))/8;return `hsl(${165+60*t},${52-12*t}%,${45-27*t}%)`;}
  function renderZone(){const f=currentFrame(),z=S.zone,selected=f.zones[z];$('tofGrid').innerHTML=f.zones.map((a,i)=>`<button class="tof-cell ${i===z?'selected':''} ${f.values[i]===null?'missing':''}" data-zone="${i}" style="background:${zoneColor(f,i)}" aria-pressed="${i===z}" aria-label="区 ${i}，测距 ${fixed(f.values[i],3)} 米，几何分数 ${fixed(a.joint,6)}" title="z${i} · ${fixed(f.values[i],3)} m · s=${fixed(a.joint,6)}">${f.values[i]===null?'—':$('heatMode').value==='score'?fixed(a.joint,2):fixed(f.values[i],1)}</button>`).join('');$('tofGrid').querySelectorAll('[data-zone]').forEach(b=>b.onclick=()=>{S.zone=Number(b.dataset.zone);S.autoZone=false;renderZone();renderPrinciple();});
    $('validZones').textContent=`有效 ${f.raw.valid_zones} / 64`;$('selectedZoneLabel').textContent=`z${z} · 第${Math.floor(z/8)+1}行 第${z%8+1}列`;$('autoZone').classList.toggle('active',S.autoZone);$('autoZone').textContent=S.autoZone?'跟随最大分数':'恢复自动跟随';$('heatLow').textContent=$('heatMode').value==='score'?'0':'0 m';$('heatHigh').textContent=$('heatMode').value==='score'?'1':'8 m';$('heatRamp').style.background=$('heatMode').value==='score'?'linear-gradient(90deg,hsl(188,25%,13%),hsl(170,60%,45%))':'linear-gradient(90deg,hsl(165,52%,45%),hsl(225,40%,18%))';
    $('zoneFacts').innerHTML=`<div><small>返回值 · 轴向深度</small><b>${fixed(f.values[z],3)} m</b></div><div><small>完整支持区间</small><b>${selected.interval?selected.interval.map(v=>fixed(v,3)).join('–')+' m':'无有效返回'}</b></div><div><small>可能相交 / 确定支持</small><b>${selected.possible?'是':'否'} / ${selected.definite?'是':'否'}</b></div><div><small>该区联合几何分数</small><b>${fixed(selected.joint,6)}</b></div>`;
    $('factorEquation').innerHTML=`深度 <b>${fixed(selected.depth,3)}</b> × 平均角域 <b>${fixed(selected.angular_given_depth,3)}</b> = <b>${fixed(selected.joint,3)}</b>`;renderOverlay();renderGeometry();renderInstruments();}
  function renderOverlay(){const f=currentFrame();let html='';if($('showGrid').checked){html=data.geometry.zone_geometry.map((g,z)=>{const [x,y,x1,y1]=g.pixel_box;return `<rect x="${x}" y="${y}" width="${x1-x}" height="${y1-y}" fill="${z===S.zone?'#f3bd6440':'transparent'}" stroke="${z===S.zone?COLORS.amber:'#ffffff48'}" stroke-width="${z===S.zone?2.5:.65}" data-overlay-zone="${z}" style="cursor:pointer"/>`;}).join('');const g=data.geometry.zone_geometry[S.zone].pixel_box;html+=`<text x="${g[0]+2}" y="${g[1]-4}" fill="${COLORS.amber}" stroke="#000" stroke-width="2" paint-order="stroke" font-size="10">z${S.zone}</text>`;}
    if($('showTruth').checked){const b=f.evaluation.target_camera_bounds_m;if(b){const focal=640/(2*Math.tan(50*Math.PI/180)),points=[];for(const x of [b.lower[0],b.upper[0]])for(const y of [b.lower[1],b.upper[1]])for(const z of [b.lower[2],b.upper[2]])if(z>0)points.push([320+focal*x/z,180+focal*y/z]);if(points.length){const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),x=Math.min(...xs),y=Math.min(...ys);html+=`<rect x="${x}" y="${y}" width="${Math.max(...xs)-x}" height="${Math.max(...ys)-y}" fill="none" stroke="${COLORS.pink}" stroke-dasharray="4 3" stroke-width="1.5"/><text x="${x}" y="${y-6}" fill="${COLORS.pink}" stroke="#000" stroke-width="2" paint-order="stroke" font-size="10">评估真值 · 未参与推理</text>`;}}}
    $('sceneOverlay').innerHTML=html;$('sceneOverlay').querySelectorAll('[data-overlay-zone]').forEach(r=>r.onclick=()=>{S.zone=Number(r.dataset.overlayZone);S.autoZone=false;renderZone();renderPrinciple();});}
  function geometryView(side){const f=currentFrame(),a=f.zones[S.zone],g=data.geometry.zone_geometry[S.zone],far=Number($('geometryRange').value),wide=far>5;const W=230,H=225,pad={l:30,r:12,t:25,b:24},pw=W-pad.l-pad.r,ph=H-pad.t-pad.b;const transverse=side?(wide?[-3.8,3.8]:[-.5,1.25]):(wide?[-3.8,3.8]:[-1.25,1.25]);
    const project=(v,z)=>side?[pad.l+z/far*pw,pad.t+(v-transverse[0])/(transverse[1]-transverse[0])*ph]:[pad.l+(v-transverse[0])/(transverse[1]-transverse[0])*pw,pad.t+(1-z/far)*ph];const pts=arr=>arr.map(p=>project(...p).map(v=>v.toFixed(2)).join(',')).join(' ');const polygon=(arr,fill,stroke,extra='')=>`<polygon points="${pts(arr)}" fill="${fill}" stroke="${stroke}" stroke-width="1.2" ${extra}/>`;
    const id=side?'sideclip':'topclip';let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${side?'侧视：Z前，Y向下':'俯视：X右，Z前'}"><defs><clipPath id="${id}"><rect x="${pad.l}" y="${pad.t}" width="${pw}" height="${ph}"/></clipPath></defs><text x="${pad.l}" y="13" fill="#afc0d3" font-size="10">${side?'侧视 · Y / Z':'俯视 · X / Z'}</text><rect x="${pad.l}" y="${pad.t}" width="${pw}" height="${ph}" rx="5" fill="#0a1420"/>`;
    for(let z=0;z<=far;z++){const p=project(transverse[0],z),q=project(transverse[1],z);svg+=`<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="#253344" stroke-width=".6"/>`;if(z===0||z===3||z===Math.floor(far))svg+=side?`<text x="${p[0]}" y="${H-9}" fill="#70849d" text-anchor="middle" font-size="8">${z}m</text>`:`<text x="${pad.l-5}" y="${p[1]+3}" fill="#70849d" text-anchor="end" font-size="8">${z}m</text>`;}
    const v=side?[-.2,.9]:[-.3,.3];svg+=`<g clip-path="url(#${id})">`+polygon([[v[0],.3],[v[1],.3],[v[1],3],[v[0],3]],'#4ed4bc10','#4ed4bc88');const mid0=project(0,0),mid1=project(0,far);svg+=`<line x1="${mid0[0]}" y1="${mid0[1]}" x2="${mid1[0]}" y2="${mid1[1]}" stroke="#576b80" stroke-dasharray="2 4" stroke-width=".7"/>`;
    if(a.interval){const [lo,hi]=a.interval,[s0,s1]=side?g.slopes_b:g.slopes_a;svg+=polygon([[s0*lo,lo],[s1*lo,lo],[s1*hi,hi],[s0*hi,hi]],'#f3bd6436',COLORS.amber);for(const s of [s0,s1]){const q=project(s*Math.min(hi,far),Math.min(hi,far));svg+=`<line x1="${mid0[0]}" y1="${mid0[1]}" x2="${q[0]}" y2="${q[1]}" stroke="#f3bd6444" stroke-dasharray="3 3" stroke-width=".7"/>`;}}
    if($('showTruth').checked){const b=f.evaluation.target_camera_bounds_m,k=side?1:0;if(b)svg+=polygon([[b.lower[k],b.lower[2]],[b.upper[k],b.lower[2]],[b.upper[k],b.upper[2]],[b.lower[k],b.upper[2]]],'none',COLORS.pink,'stroke-dasharray="3 3"');}
    svg+='</g>';const cam=project(0,0);svg+=`<circle cx="${cam[0]}" cy="${cam[1]}" r="3" fill="#c9d8e8"/><text x="${side?pad.l+pw-28:pad.l+pw-22}" y="${side?H-9:H-9}" fill="#a0b2c8" font-size="9">${side?'Z →':'X →'}</text><text x="${side?6:pad.l-21}" y="${side?pad.t+ph-4:pad.t+10}" fill="#a0b2c8" font-size="9">${side?'Y↓':'Z↑'}</text>`;
    if(!a.interval||a.interval[0]>far)svg+=`<text x="${pad.l+pw/2}" y="${pad.t+ph*.5}" fill="#9baec4" font-size="9" text-anchor="middle">${a.interval?'支持区间在视窗外':'该区无有效返回'}</text>`;svg+='</svg>';return svg;}
  function renderGeometry(){$('geometryViews').innerHTML=geometryView(false)+geometryView(true);}
  function renderTimeline(){const clip=currentClip(),frames=clip.frames,f=currentFrame(),W=480,H=172,L=55,R=22,T=15,B=29,pw=W-L-R,ph=H-T-B,x=i=>L+i/Math.max(1,frames.length-1)*pw,y=s=>T+(1-s)*ph;let svg='';
    for(const s of [0,.5,1])svg+=`<line x1="${L}" y1="${y(s)}" x2="${W-R}" y2="${y(s)}" stroke="#2c394b" stroke-width=".7"/><text x="${L-10}" y="${y(s)+3}" text-anchor="end" fill="#8093aa" font-size="9">${s.toFixed(1)}</text>`;
    const th=data.thresholds.strong;svg+=`<line x1="${L}" y1="${y(th)}" x2="${W-R}" y2="${y(th)}" stroke="${COLORS.amber}" stroke-width="1" stroke-dasharray="4 4"/><text x="${W-R}" y="${y(th)-6}" text-anchor="end" fill="${COLORS.amber}" font-size="9">T = 0.407131</text>`;
    svg+=`<polyline points="${frames.map((r,i)=>x(i)+','+y(r.score)).join(' ')}" fill="none" stroke="${COLORS.teal}" stroke-width="2"/>`;
    svg+=frames.map((r,i)=>`<circle cx="${x(i)}" cy="${y(r.score)}" r="${i===S.frame?4:2.2}" fill="${i===S.frame?COLORS.amber:COLORS.teal}"/>`).join('');svg+=`<line x1="${x(S.frame)}" y1="${T}" x2="${x(S.frame)}" y2="${H-B}" stroke="#e2edf5" stroke-width=".9"/>`;
    for(let i=0;i<frames.length;i+=Math.max(1,Math.ceil(frames.length/7)))svg+=`<text x="${x(i)}" y="${H-10}" text-anchor="middle" fill="#8b9bb0" font-size="9">${fixed(frames[i].time_s,1)}s</text>`;
    $('scoreTimeline').innerHTML=svg;$('scoreTimeline').onclick=e=>{const r=e.currentTarget.getBoundingClientRect(),q=(e.clientX-r.left)/r.width*W;setFrame((q-L)/pw*(frames.length-1));};$('clipDuration').textContent=frames.length+' 个采样';
    const row=(title,kind)=>`<div class="track" style="--samples:${frames.length}"><span>${title}</span>${frames.map((r,i)=>{let color='#253243',label='';if(kind==='truth'){color=r.evaluation.truth?'#8b74aa':'#253243';label=!hasTruth(r)?'未标注':r.evaluation.truth?'正例':'负例';}else{const a=flagFor(r,kind),truth=r.evaluation.truth;const outcome=outcomeFor(truth,a);color=outcome==='FP'?COLORS.red:outcome==='FN'?COLORS.pink:a?(kind==='hold'&&r.decision.held_only?COLORS.amber:COLORS.teal):'#253243';label=(outcome==='UNLABELED'?(a?'提醒 · 未评价':'未提醒 · 未评价'):outcome)+(kind==='hold'&&r.decision.held_only?' · 保持':'');}return `<button data-timeline-frame="${i}" class="${i===S.frame?'current':''}" style="background:${color}" title="${title} · ${r.id} · ${label}" aria-label="${title}，${r.id}，${label}"></button>`;}).join('')}</div>`;
    $('decisionTracks').innerHTML=row('评估真值','truth')+row('Cal.','calibration')+row('Strong','strong')+row('候选','hold');$('decisionTracks').querySelectorAll('[data-timeline-frame]').forEach(b=>b.onclick=()=>setFrame(Number(b.dataset.timelineFrame)));
    const q=clipStats(clip);$('timelineLegend').classList.toggle('unlabeled',!frames.some(hasTruth));$('clipSummary').innerHTML=!frames.some(hasTruth)?`<span>评估状态<b>未标注 · 不计算准确率</b></span><span>当前策略提醒<b>${frames.filter(r=>r.decision.alert).length} 帧</b></span><span>仅保持提醒<b>${frames.filter(r=>r.decision.held_only).length} 帧</b></span>`:`<span>本片段正帧覆盖<b>${q.tp} / ${q.tp+q.fn}</b></span><span>本片段误报<b>${q.fp} 帧 · ${q.segments} 段</b></span><span>保持新增提醒<b>${frames.filter(r=>r.decision.held_only).length} 帧</b></span>`;}
  function renderMechanism(){if(S.page!=='principle')return;window.Mechanism.render({frame:currentFrame(),clip:currentClip(),index:S.frame,zone:S.zone,geometry:data.geometry,threshold:data.thresholds.strong,title:clipTitle(currentClip()),onZone:z=>{S.zone=z;S.autoZone=false;renderZone();renderPrinciple();},onSeek:i=>setFrame(i)});}
  function renderPrinciple(){renderMechanism();const f=currentFrame(),a=f.zones[S.zone],d=f.decision;$('liveFactor').innerHTML=`当前选中 <b>z${S.zone}</b>：深度占比 <b>${fixed(a.depth,6)}</b> × 区间内平均角域重叠 <b>${fixed(a.angular_given_depth,6)}</b> = 联合分数 <b>${fixed(a.joint,6)}</b>。<br>全帧最大分数为 <b>${fixed(f.score,6)}</b>。${a.interval?'该区完整测距区间 '+a.interval.map(v=>fixed(v,3)).join('–')+' m。':'该区无有效返回。'}`;$('liveState').innerHTML=`当前 ${esc(f.id)} · ${fixed(f.time_s,1)} s：<br>当前强证据 <b>${d.strong?'有':'无'}</b>；上一相邻采样强证据 <b>${d.previous_strong?'有':'无'}</b>。<br>最终：<b>${d.held_only?'仅保持一帧':d.alert?'当前证据提醒':'不触发提醒'}</b>。空间 UNKNOWN：${d.unknown?'保留':'当前有确定支持'}。`;}
  function renderResults(){const scope=$('resultScope').value,names={core:'完整 Core 布局',boundary:'Boundary 挑战',all:'全部布局'};$('resultCards').innerHTML=evaluatedCohorts(data.cohorts).map(co=>{const m=co.metrics[scope],h=m.hold;return `<article class="panel result-card"><div class="result-heading"><div><h2>${esc(co.title)}</h2><p>${esc(co.subtitle)}</p></div><span class="tag ${scope==='boundary'?'pink':'neutral'}">${names[scope]} · ${h.frames}帧 · ${h.positive}P / ${h.negative}N</span></div><div class="table-scroll"><table><thead><tr><th>固定读出</th><th>TP / FP / FN</th><th>精确率</th><th>召回率</th><th>F1</th><th>FPR</th><th>事件</th><th>误报段 / 时长</th><th>最大首报延迟</th></tr></thead><tbody>${['calibration','strong','hold'].map(arm=>{const a=m[arm];return `<tr class="${arm==='hold'?'highlight':''}"><td>${arm==='calibration'?'Calibration':arm==='strong'?'固定强阈值':'强阈值 + 一帧保持'}</td><td>${a.TP} / ${a.FP} / ${a.FN}</td><td>${pct(a.precision,2)}</td><td>${pct(a.recall,2)}</td><td>${pct(a.f1,2)}</td><td>${pct(a.FPR,2)}</td><td>${a.events_detected} / ${a.event_count}</td><td>${a.false_segments} / ${fixed(a.false_sampled_s,1)} s</td><td>${fixed(a.first_alert_max_delay_s,1)} s</td></tr>`;}).join('')}</tbody></table></div><div class="result-detail">候选入界前已在报警：${h.preexisting_alert_events} 个事件；当前空间 UNKNOWN：${h.prediction_unknown}/${h.frames} 帧。最大延迟仅统计已检出事件，漏事件数为 ${h.event_count-h.events_detected}。<br>来源：${esc(co.source)} · 每个布局完整保留，未按告警结果删帧。</div></article>`;}).join('');}
  async function guide(kind){stop();let ci=kind==='delay'?data.cohorts.findIndex(c=>c.id==='transfer'):data.cohorts.findIndex(c=>c.id==='validation');if(ci<0)ci=0;const co=data.cohorts[ci];let found=null;for(let i=0;i<co.clips.length&&!found;i++){const c=co.clips[i];for(let j=0;j<c.frames.length;j++){const f=c.frames[j];const match=kind==='hold'?c.layout==='INSIDE'&&f.decision.held_only:kind==='outside'?c.layout==='OUTSIDE'&&f.decision.calibration&&!f.decision.alert:kind==='delay'?c.layout==='INSIDE'&&f.evaluation.truth&&!f.decision.alert&&j>0&&!c.frames[j-1].evaluation.truth:c.layout==='BOUNDARY'&&f.evaluation.truth&&!f.decision.alert;if(match){found=[i,j];break;}}}if(found){if(!await selectCohort(ci,...found))return;setPage('replay');const label={hold:'一帧保持补回提醒；当前测量没有改变',outside:'Calibration 报警，固定候选抑制了该负帧',delay:'旧数据中的首报延迟仍然保留',boundary:'边界挑战的漏报也完整展示'}[kind];$('guideNote').textContent=label;toast(label);}else toast('本数据集中没有该类型示例。');}
  $('cohort').innerHTML=data.cohorts.map((c,i)=>`<option value="${i}">${esc(c.title)}</option>`).join('');$('cohort').onchange=e=>selectCohort(Number(e.target.value));$('familyFilter').onchange=e=>{S.family=e.target.value;renderSceneList();renderGallery();};document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{S.layout=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(q=>q.classList.toggle('active',q===b));renderSceneList();renderGallery();});document.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>setPage(b.dataset.page));document.querySelectorAll('[data-goto]').forEach(b=>b.onclick=()=>setPage(b.dataset.goto));document.querySelectorAll('[data-guide]').forEach(b=>b.onclick=()=>guide(b.dataset.guide));document.querySelector('.brand').onclick=e=>{e.preventDefault();setPage('gallery');};
  $('mechanismPrev').onclick=()=>setFrame(S.frame-1);$('mechanismNext').onclick=()=>setFrame(S.frame+1);$('mechanismSeek').oninput=e=>setFrame(Number(e.target.value));
  document.querySelectorAll('[data-mechanism-example]').forEach(b=>b.onclick=()=>{const kind=b.dataset.mechanismExample;const i=currentClip().frames.findIndex(f=>kind==='strong'?f.decision.strong:kind==='hold'?f.decision.held_only:!f.decision.alert);if(i>=0)setFrame(i);else toast('当前片段没有该状态，请从目录选择其他场景。');});
  $('catalogToggle').onclick=()=>{const open=document.body.classList.toggle('catalog-open');$('catalogToggle').setAttribute('aria-expanded',String(open));if(open){document.body.classList.remove('wide');$('presentation').textContent='放大视图';}};
  $('startTour').onclick=startTour;$('stopTour').onclick=()=>{stop();S.tour=null;renderTour();};
  $('play').onclick=play;$('prevFrame').onclick=()=>setFrame(S.frame-1);$('nextFrame').onclick=()=>setFrame(S.frame+1);$('seek').oninput=e=>setFrame(Number(e.target.value));$('speed').onchange=()=>{if(S.playing)schedule();};$('heatMode').onchange=renderZone;$('geometryRange').onchange=renderGeometry;$('autoZone').onclick=()=>{S.autoZone=true;S.zone=dominantZone(currentFrame());renderZone();renderPrinciple();};$('showGrid').onchange=renderOverlay;$('showTruth').onchange=()=>{renderOverlay();renderGeometry();};$('resultScope').onchange=renderResults;$('presentation').onclick=()=>{const wide=document.body.classList.toggle('wide');const catalog=!wide&&window.matchMedia('(max-width:980px)').matches;document.body.classList.toggle('catalog-open',catalog);$('catalogToggle').setAttribute('aria-expanded',String(catalog));$('presentation').textContent=wide?'显示目录':'放大视图';};
  document.addEventListener('keydown',e=>{if(S.page!=='replay'||['INPUT','SELECT','TEXTAREA','BUTTON'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();play();}else if(e.code==='ArrowRight'){e.preventDefault();setFrame(S.frame+1);}else if(e.code==='ArrowLeft'){e.preventDefault();setFrame(S.frame-1);}});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  setPage('gallery');selectCohort(extra&&extra.clips&&extra.clips.length?data.cohorts.length-1:0);
})();
