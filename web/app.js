import {Playback} from './playback.mjs';
import {GenerationVisualizer} from './generation-view.mjs';
import {GenerationCoverage,generationTaskCounts,generationTaskLabel} from './generation-coverage.mjs';
import {ObjectsView} from './objects-view.mjs';
import {ImportView} from './import-view.mjs';
import {ReplayDatasetBuilder} from './replay-builder.mjs';
import {ObjectZonePicker,ZONE_OBJECTS,LEGACY_EXTRA_OBJECTS,activeZoneObjects,zoneSummary} from './generation-zones.mjs';
const $=s=>document.querySelector(s), $$=s=>document.querySelectorAll(s);
// Simulation appearance is display only: archived states are re-rendered, never re-simulated.
const APPEARANCES={raw:{label:'MUJOCO',name:'Raw'},background:{label:'REAL BG',name:'Real background'},realistic:{label:'REALISTIC',name:'Realistic'}};
function savedAppearance(){try{const value=localStorage.getItem('sim-appearance');return APPEARANCES[value]?value:'raw';}catch{return 'raw';}}
const importedDataset=()=>state.datasets?.find(d=>d.id===state.selectedDataset)?.source==='import';
const appearanceQuery=(appearance=state.appearance)=>{const parts=[];if(appearance!=='raw')parts.push('appearance='+appearance);if(importedDataset())parts.push('dataset='+encodeURIComponent(state.selectedDataset));return parts.length?'?'+parts.join('&'):'';};
const episodeVideo=()=>state.episode?.videos?.[state.appearance]||null;
const clientId=crypto.randomUUID();
const state={token:'',runs:[],historyMode:false,page:'datasets',datasets:[],selectedDataset:null,restarting:false,instance:null,activeCapture:null,episodes:[],episode:null,index:0,frame:0,view:'top',mode:'dataset',capture:null,selection:null,selectionRole:'from',selecting:false,plan:null,robot:{},busy:false,filter:'all',lastJob:'',request:0,appearance:savedAppearance()};
const video=$('#video'),real=$('#real-canvas'),sim=$('#sim-canvas');
const playback=new Playback(video,{onFrame:()=>{draw();if(state.mode==='dataset'&&state.episode)state.frame=Math.min(state.episode.frames-1,Math.floor(video.currentTime*30));updateTimeline();},onState:({ready,paused})=>{$('#play').disabled=!ready||state.robot.running||(state.mode==='live'&&state.plan?.executed&&!state.plan?.execution?.video);$('#play').textContent=paused?'▶':'Ⅱ';$('#play').setAttribute('aria-label',paused?'Play comparison':'Pause comparison');},onError:message=>toast(message,true)});
function planVideo(){const name=state.plan?.execution?.video||state.plan?.video;return name?`/media/studio/${state.capture.id}/${name}`:null;}
function playbackLabel(message){$('#playback-status').textContent=message||((state.mode==='dataset'||state.plan?.execution?.video)?'⛓ Same timeline':'Capture still + sim video');}

let masks={},combined=null,realImage=null,simImage=null,toastTimer,seekTimer,liveBusy=false;
const pretty=s=>String(s||'').replaceAll('_',' ');
function event(message){const li=document.createElement('li'),t=document.createElement('time'),s=document.createElement('span');t.textContent=new Date().toLocaleTimeString('en-GB');s.textContent=message;li.append(t,s);$('#events').prepend(li);while($('#events').children.length>12)$('#events').lastChild.remove();$('#activity-status').textContent=message;}
function toast(message,error=false){$('#toast').textContent=message;$('#toast').className='toast'+(error?' error':'');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').classList.add('hidden'),6500);if(error)event(message);}
async function api(path,body,upload){const options=body===undefined?{headers:{'X-Studio-Client':clientId}}:{method:'POST',headers:{'Content-Type':upload?body.type:'application/json','X-Studio-Token':state.token,'X-Studio-Client':clientId,...upload?.headers},body:upload?body:JSON.stringify(body)};const response=await fetch(path,options);const result=await response.json();if(!response.ok)throw Error(result.error||result.detail||'Request failed');return result;}
function image(url){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(Error('Image unavailable'));img.src=url;});}
function blank(which,text){$('#'+which+'-empty').textContent=text;$('#'+which+'-empty').classList.remove('hidden');}
function draw(){
 const r=real.getContext('2d'),s=sim.getContext('2d');
 if(state.mode==='dataset'){
  const source=episodeVideo()&&video.readyState>=2&&video.videoHeight===1024?video:combined;
  if(source){let y=state.view==='top'?32:544;r.drawImage(source,0,y,640,480,0,0,640,480);s.drawImage(source,640,y,640,480,0,0,640,480);$('#real-empty').classList.add('hidden');$('#sim-empty').classList.add('hidden');}
 }else{
  if(state.plan?.execution?.video&&video.readyState>=2&&video.videoHeight===1024){
    const y=state.view==='top'?32:544;r.drawImage(video,0,y,640,480,0,0,640,480);s.drawImage(video,640,y,640,480,0,0,640,480);$('#real-empty').classList.add('hidden');$('#sim-empty').classList.add('hidden');return;
  }
  if(realImage){r.drawImage(realImage,0,0,640,480);$('#real-empty').classList.add('hidden');if(state.view==='top'&&!state.robot.running&&!state.plan?.executed&&state.capture?.source==='live'){if(state.capture?.task==='pickplace'){if(state.selection?.objects)pickplaceDraw(r);}else{for(const role of ['from','to'])if(masks[role])r.drawImage(masks[role],0,0,640,480);for(const m of masks.obstacles||[])r.drawImage(m,0,0,640,480);for(const [role,item,color] of [['from',state.selection?.from,'#328cd2'],['to',state.selection?.to,'#327b42'],...(state.selection?.obstacles||[]).map(o=>['obstacle',o,'#d24a3c'])]){if(!item)continue;r.fillStyle=color;r.font=role==='obstacle'?'bold 12px sans-serif':'bold 16px sans-serif';const at=item.bbox?[item.bbox[0],Math.max(20,item.bbox[1]-8)]:[item.point[0]-14,Math.max(20,item.point[1]-16)];r.fillText(role==='obstacle'?'OBSTACLE':role==='to'&&item.kind==='place'?'TO · EMPTY SPOT':role.toUpperCase(),...at);r.strokeStyle=color;r.lineWidth=2;r.beginPath();r.arc(...item.point,item.kind==='place'?12:5,0,Math.PI*2);r.stroke();}}}}
  if(state.plan?.video&&video.readyState>=2&&video.videoHeight===480){s.drawImage(video,state.view==='top'?0:640,0,640,480,0,0,640,480);$('#sim-empty').classList.add('hidden');}
  else if(simImage&&state.view==='top'){s.drawImage(simImage,0,0,640,480);$('#sim-empty').classList.add('hidden');}
 }
}
function renderList(){if(state.historyMode){renderRuns();return;}const query=$('#search').value.toLowerCase();$('#episodes').replaceChildren();for(const e of state.episodes){if(state.filter==='failed'&&e.outcome==='success')continue;if(query&&!String(e.episode).padStart(2,'0').includes(query)&&!`episode ${e.episode}`.includes(query))continue;const b=document.createElement('button');b.className='episode-item'+(e.episode===state.index?' active':'');b.innerHTML=`<span class="ep-result ${e.outcome==='success'?'':'fail'}">${e.outcome==='success'?'✓':'•'}</span><span class="ep-num">Episode ${String(e.episode).padStart(2,'0')}</span><span class="ep-time">${(e.frames/30).toFixed(1)} s</span>`;b.onclick=()=>loadEpisode(e.episode);$('#episodes').append(b);}}
function setMode(mode){
 if(mode!==state.mode)state.request++;video.pause();state.mode=mode;
 $('#real-label').textContent=mode==='dataset'?'RECORDED':state.plan?.execution?.video?'RECORDED RUN':'CAPTURE STILL';
 $('#sim-label').textContent=mode==='dataset'?APPEARANCES[state.appearance].label:'PLANNED';$('#sim-appearance').classList.toggle('hidden',mode!=='dataset');
 $('#comparison-note').textContent=mode==='dataset'?'Original recorded actions · shared robot, camera and contact parameters':'Image-conditioned IK plan · simulation success does not establish real-world success';
 if(mode==='dataset'&&state.episode){setVideo(episodeVideo());$('#episode-label').textContent=`EPISODE ${String(state.index).padStart(2,'0')}`;setOutcome(state.episode.report.outcome);}
 else{setVideo(planVideo());$('#episode-label').textContent=state.page==='history'?'RECORDED RUN':'NEW RUN';setOutcome(state.plan?.outcome||(state.capture?'select objects':'capture needed'));if(!realImage)blank('real','Connect the robot and capture the current scene.');if(!simImage&&!state.plan?.video)blank('sim','Generate a plan to reconstruct this scene.');}
 applyLayout();draw();chart();updateTimeline();playbackLabel();updateSelection();
}
function setOutcome(text){$('#outcome').textContent=(state.mode==='live'&&text==='success'?'sim: ':'')+pretty(text);$('#outcome').classList.toggle('failed',!['success','not_simulated'].includes(text));}
function setVideo(url){playback.setSource(url);playbackLabel();}
async function ensureEpisodeVideo(index,request,appearance=state.appearance){
 const stale=()=>request!==state.request||state.mode!=='dataset'||state.index!==index||state.appearance!==appearance;
 try{
  let job=await api(`/api/episodes/${index}/video${appearanceQuery(appearance)}`,{});
  while(job.status==='rendering'){
   if(stale())return;
   playbackLabel(appearance==='raw'?'Preparing full video…':`Rendering ${APPEARANCES[appearance].name.toLowerCase()} video…`);
   await new Promise(resolve=>setTimeout(resolve,1000));
   job=await api(`/api/episodes/${index}/video${appearanceQuery(appearance)}`);
  }
  if(stale())return;
  if(job.status!=='ready')throw Error(job.error||'Could not prepare video');
  state.episode.videos[appearance]=job.video;showEpisodeVideo(job.video,state.frame);playbackLabel('⛓ Same timeline');
  event(`Episode ${String(index).padStart(2,'0')} ${appearance==='raw'?'full':APPEARANCES[appearance].name.toLowerCase()} video ready · both views use one clock`);
 }catch(e){if(!stale()){playbackLabel('Video unavailable');toast(e.message,true);}}
}
function showEpisodeVideo(url,frame){setVideo(url);if(frame>0)video.addEventListener('loadeddata',()=>{if(video.getAttribute('src')===url)video.currentTime=frame/30;},{once:true});}
async function changeAppearance(){
 const appearance=$('#sim-appearance').value;if(!APPEARANCES[appearance]||appearance===state.appearance)return;
 state.appearance=appearance;try{localStorage.setItem('sim-appearance',appearance);}catch{}
 $('#sim-label').textContent=state.mode==='dataset'?APPEARANCES[appearance].label:'PLANNED';
 if(state.mode!=='dataset'||!state.episode)return;
 const request=++state.request,index=state.index,frame=state.frame,url=episodeVideo();
 if(url){showEpisodeVideo(url,frame);event(`Simulation view · ${APPEARANCES[appearance].name}`);return;}
 setVideo(null);combined=null;blank('sim',`Rendering ${APPEARANCES[appearance].name.toLowerCase()} view…`);
 ensureEpisodeVideo(index,request,appearance);
 try{const still=await image(`/api/episodes/${index}/frame/${frame}${appearanceQuery(appearance)}`);if(request!==state.request||state.mode!=='dataset'||state.index!==index||state.frame!==frame||episodeVideo())return;combined=still;draw();}catch(e){toast(e.message,true);}
}
async function loadEpisode(index){try{state.page='datasets';state.historyMode=false;historySidebar();$('#run-summary').classList.add('hidden');if(state.mode!=='dataset'){state.mode='dataset';state.request++;}const request=++state.request;video.pause();blank('real','Loading episode…');blank('sim','Loading synchronized simulation…');const result=await api(`/api/episodes/${index}${appearanceQuery('raw')}`);if(request!==state.request)return;state.index=index;$('#episode-select').value=index;state.episode=result;result.videos={raw:result.video||null};state.frame=0;combined=null;setMode('dataset');if(episodeVideo()&&video.readyState>=2)video.currentTime=0;renderList();$('#timeline').max=result.frames-1;$('#timeline').value=0;const rmse=result.report.joint_state_rmse_native;$('#arm-error').innerHTML=`${Math.sqrt(rmse.slice(0,5).reduce((a,x)=>a+x*x,0)/5).toFixed(3)}<em>°</em>`;$('#gripper-error').innerHTML=`${rmse[5].toFixed(3)}<em>u</em>`;if(!episodeVideo()){ensureEpisodeVideo(index,request);const firstFrame=await image(`/api/episodes/${index}/frame/0${appearanceQuery()}`);if(request!==state.request||state.mode!=='dataset'||state.index!==index)return;combined=firstFrame;draw();}event(`Episode ${String(index).padStart(2,'0')} loaded · ${pretty(result.report.outcome)}${episodeVideo()?' · synchronized video':' · frame-by-frame view'}`);}catch(e){toast(e.message,true);}}
function updateTimeline(){const duration=state.mode==='dataset'?state.episode?.duration||0:state.plan?.execution?.duration||state.plan?.duration||0;const now=state.mode==='dataset'?state.frame/30:video.currentTime||0;$('#time').textContent=`${now.toFixed(2)} / ${duration.toFixed(2)} s`;const count=state.mode==='dataset'?state.episode?.frames||1:state.plan?.execution?.frames||state.plan?.frames||1;$('#timeline').max=count-1;$('#timeline').value=Math.min(count-1,Math.round(now*30));$('#frame-number').textContent=`FRAME ${String(Math.min(count-1,Math.round(now*30))).padStart(3,'0')} / ${String(count-1).padStart(3,'0')}`;const cursor=$('#chart-cursor');if(cursor)cursor.setAttribute('x1',45+(Math.min(count-1,Math.round(now*30))/Math.max(1,count-1))*740),cursor.setAttribute('x2',cursor.getAttribute('x1'));}
function chart(){const svg=$('#chart');svg.replaceChildren();if(state.mode!=='dataset'||!state.episode){svg.innerHTML='<text x="400" y="72" text-anchor="middle" fill="#9ba991" font-size="11">Select a recorded episode to compare joint tracking.</text>';return;}const j=Number($('#joint').value||0),a=state.episode.real.map(x=>x[j]),b=state.episode.sim.map(x=>x[j]);let low=Math.min(...a,...b),high=Math.max(...a,...b);const margin=Math.max(2,(high-low)*.12);low-=margin;high+=margin;const x=i=>45+i/(a.length-1)*740,y=v=>112-(v-low)/(high-low)*95;let html='';for(let i=0;i<4;i++){let v=low+(high-low)*i/3;html+=`<line x1="45" y1="${y(v)}" x2="785" y2="${y(v)}" stroke="#e9eee4" stroke-dasharray="3 5"/><text x="34" y="${y(v)+3}" text-anchor="end" font-size="8" fill="#a1ac97">${v.toFixed(0)}</text>`;}for(let i=0;i<5;i++){let at=i/4;html+=`<text x="${45+at*740}" y="134" font-size="8" text-anchor="middle" fill="#a1ac97">${(at*state.episode.duration).toFixed(1)}s</text>`;}for(const [values,color,dash] of [[a,'#4689b1',''],[b,'#9475b9','4 3']])html+=`<polyline points="${values.map((v,i)=>`${x(i)},${y(v)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="1.8" stroke-dasharray="${dash}"/>`;html+='<line id="chart-cursor" x1="45" x2="45" y1="10" y2="116" stroke="#acc098" stroke-width="1"/>';svg.innerHTML=html;$('#chart-unit').textContent=j===5?'Gripper · normalized 0–100 units':'Joint position · degrees';updateTimeline();}
const motionLabels={natural:'Natural',classic:'Classic',gentle:'Gentle',flow:'Flow'};
const motionTimingLabels={recorded:'Recorded robot pace',conservative:'Conservative studio pace'};
function motionMismatch(){return !!state.plan&&($('#motion-technique').value!==(state.plan.motion_technique||'classic')||Number($('#motion-speed').value)/100!==(state.plan.speed_scale??1)||$('#motion-timing').value!==(state.plan.timing_profile||'conservative'));}
function updateMotion(){
 const changed=motionMismatch(),percent=Number($('#motion-speed').value),recorded=$('#motion-timing').value==='recorded';
 $('#motion-speed-value').textContent=percent+'%';
 if(recorded){
  const duration=!changed&&Number.isFinite(state.plan?.duration)?`Plan duration: ${state.plan.duration.toFixed(1)} s.`:'Generate a plan to see its actual duration.';
  $('#motion-speed-note').textContent=`Recorded pace uses per-joint speed caps; tracking limits can cap the requested speed. ${duration} Simulation and the robot use this same planned timing.`;
 }else{
  const cap=35*percent/100*($('#motion-technique').value==='classic'?.8:.92);
  $('#motion-speed-note').textContent=`Arm speed cap: ${cap.toFixed(1)}°/s per joint. Gripper speed and grasp/release waits retained. `+($('#motion-technique').value==='classic'?'Original stop-at-each-point motion.':'Acceleration and jerk limits unchanged.');
 }
 const descriptions={natural:'Coordinated curved travel with a short centered descent and settling before grasp. Controlled stops for release.',classic:'A full stop at each waypoint, using the selected motion timing.',gentle:'Soft quintic acceleration and deceleration. Blends travel points; may take longer.',flow:'Continuous travel with bounded acceleration and jerk. Controlled stops for grasp and release.'};
 $('#motion-note').textContent=changed?'Motion settings changed. Generate a new plan and run simulation.':descriptions[$('#motion-technique').value];
 $('#motion-note').classList.toggle('changed',changed);
}
function updateControls(){$('#home').disabled=state.page!=='new'||state.busy||state.selecting||state.robot.running||!state.robot.connected||state.robot.torque_enabled!==true;$('#torque-off').disabled=state.page!=='new'||state.busy||state.selecting||state.robot.running||!state.robot.connected||state.robot.torque_enabled===false;$('#torque-on').disabled=state.page!=='new'||state.busy||state.selecting||state.robot.running||!state.robot.connected||state.robot.torque_enabled===true;$('#prepare-state').textContent=state.robot.running?'MOVING':!state.robot.connected?'CONNECT':state.robot.torque_enabled===false?'TORQUE OFF':state.robot.torque_enabled===true?'READY':'TORQUE UNKNOWN';$('#speed').disabled=!!state.robot.running;$('#timeline').disabled=!!state.robot.running;$('#play').disabled=!playback.ready||!!state.robot.running||(state.mode==='live'&&state.plan?.executed&&!state.plan?.execution?.video);const busy=state.busy||state.selecting||state.robot.running||state.restarting;$('#task').disabled=busy;$('#motion-technique').disabled=busy||state.page!=='new'||!!state.plan?.executed;$('#motion-speed').disabled=$('#motion-technique').disabled;$('#motion-timing').disabled=$('#motion-technique').disabled;updateMotion();const cap=state.capture,plan=state.plan,archive=state.page!=='new'||!!cap&&cap.id!==state.activeCapture;for(const id of ['nav-data','nav-live','nav-history','nav-generate','back-datasets'])$('#'+id).disabled=busy;$('#restart-server').disabled=state.restarting;$('#run-select').disabled=busy;$('#capture').disabled=state.page!=='new'||busy||($('#source').value==='live'&&(!state.robot.connected||state.robot.torque_enabled!==true));$('#plan').disabled=busy||archive||!cap||cap?.source==='live'&&!selectionRoles().every(role=>state.selection?.[role])||!!plan?.executed;$('#simulate').disabled=busy||archive||!plan||plan.executed||motionMismatch();$('#execute').disabled=busy||archive||motionMismatch()||!state.robot.connected||!plan||plan.source!=='live'||plan.outcome!=='success'||!plan.transfer_checks_passed||plan.executed||Date.now()/1000-plan.captured_at>180;$('#connect').disabled=state.busy||state.selecting||state.robot.running;$('#connect').textContent=state.robot.connected?'Disconnect':'Connect robot';$('#capture-state').textContent=cap?'CAPTURED':'READY';$('#plan-state').textContent=plan?'READY':'WAITING';$('#simulate-state').textContent=plan&&plan.outcome!=='not_simulated'?plan.outcome==='success'?'PASSED':'FAILED':'WAITING';if(plan?.source==='live'&&!plan.executed&&Date.now()/1000-plan.captured_at>180)$('#pipeline-hint').textContent='Capture expired. Capture, generate a plan, and simulate again before execution.';$('#execute-state').textContent=state.robot.running?`${Math.round(state.robot.progress*100)}%`:plan?.executed?'CONSUMED':$('#execute').disabled?'LOCKED':'READY';for(const step of ['capture','plan','simulate','execute'])$('#step-'+step).classList.toggle('done',step==='capture'?!!cap:step==='plan'?!!plan:step==='simulate'?plan?.outcome==='success':plan?.executed&&!state.robot.running);const all=$('#run-all');all.disabled=state.runAll||state.page!=='new'||busy||$('#source').value!=='live'||!state.robot.connected||state.robot.torque_enabled!==true;$('#calibrate-camera').disabled=all.disabled;all.textContent=state.runAll?'Running all steps…':'▶▶  Run all';if(state.runAll)for(const id of ['capture','plan','simulate','execute','task','motion-technique','motion-speed','motion-timing','home','torque-off','torque-on'])$('#'+id).disabled=true;updateSelection();}
function sourceChanged(){$('#task-note').textContent='One task for every object: SAM finds everything on the mat, you click From (the object to pick) and To (another object, the bowl, or an empty spot). Stacking, unstacking, into or out of the bowl and moving the bowl by its rim all use the same plan, from each object\'s contact areas (3D-Objects page).';$('#capture').textContent='◎  Capture current scene';$('#pipeline-hint').textContent='Every detected object is simulated as a real body. Simulation and a fresh scene check are required before execution.';updateControls();}
async function capturedImage(){if(!state.capture||state.robot.running||state.plan?.execution?.video||(state.page==='new'&&state.plan?.executed&&state.capture.id===state.activeCapture))return;const id=state.capture.id,view=state.view;const frame=await image(`/media/studio/${id}/${view}.jpg`);if(state.mode!=='live'||state.capture?.id!==id||state.view!==view||state.robot.running||state.plan?.execution?.video||(state.page==='new'&&state.plan?.executed&&state.capture.id===state.activeCapture))return;realImage=frame;draw();}
async function showPlan(plan){const id=state.capture?.id,request=state.request;state.plan=plan;$('#motion-technique').value=plan.motion_technique||'classic';$('#motion-timing').value=plan.timing_profile||'conservative';$('#motion-speed').value=String(Math.round((plan.speed_scale??1)*100));const block=$('#plan-result');block.classList.remove('hidden');block.replaceChildren();const title=document.createElement('strong');title.textContent=plan.outcome==='not_simulated'?'Plan ready for physics test':`Simulation: ${pretty(plan.outcome)}`;const details=document.createElement('div');details.textContent=`${motionLabels[plan.motion_technique||'classic']} · ${motionTimingLabels[plan.timing_profile||'conservative']} · ${Math.round((plan.speed_scale??1)*100)}% travel · ${plan.frames} actions · ${plan.duration.toFixed(1)} s · IK ${plan.max_ik_error_mm.toFixed(1)} mm`;block.append(title,details);if(plan.motion&&plan.motion_technique!=='classic'){const timing=document.createElement('div');timing.textContent=`Conservative Classic at 100% estimate: ${plan.motion.classic_duration.toFixed(1)} s · ${Math.abs(plan.motion.time_saved_seconds).toFixed(1)} s ${plan.motion.time_saved_seconds>=0?'shorter':'longer'}`;block.append(timing);}if(plan.outcome!=='not_simulated'){const safety=document.createElement('div');safety.textContent=plan.transfer_checks_passed?'Transfer checks passed':'Transfer checks did not pass — hardware run locked'+(plan.transfer_check_failures?.length?': '+plan.transfer_check_failures.join('; '):'');block.append(safety);}const link=document.createElement('a');link.href=`/media/studio/${state.capture.id}/plan.json`;link.textContent='View plan & simulation report ↗';link.target='_blank';link.rel='noopener';block.append(link);const replay=document.createElement('a');replay.classList.toggle('hidden',!plan.executed);replay.href='/?run='+state.capture.id;replay.textContent='Open this run ↗';block.append(document.createElement('br'),replay);try{const preview=plan.purpose==='home'?null:await image(`/media/studio/${id}/preview.jpg`);if(state.capture?.id!==id||state.request!==request)return;simImage=preview;}catch{}if(state.capture?.id!==id||state.request!==request)return;if(state.mode==='live'){setOutcome(plan.outcome);if(plan.video||plan.execution?.video)setVideo(planVideo());$('#real-label').textContent=plan.execution?.video?'RECORDED RUN':'CAPTURE STILL';playbackLabel();draw();}updateTimeline();updateControls();}
async function doCapture(){let ok=false;try{state.request++;state.busy=true;updateControls();const result=await api('/api/capture',{source:$('#source').value,episode:state.index,task:$('#task').value});state.page='new';state.historyMode=false;state.capture=result.capture;state.activeCapture=state.capture.id;$('#run-summary').classList.add('hidden');history.replaceState(null,'','?run='+state.capture.id);state.selection=null;state.selectionRole='from';masks={};state.plan=null;realImage=null;simImage=null;$('#plan-result').classList.add('hidden');setMode('live');await capturedImage();event(`${result.capture.source==='live'?'Live scene':'Episode '+state.index+' rehearsal'} captured · six native joint positions saved`);ok=true;}catch(e){toast(e.message,true);}finally{state.busy=false;updateControls();}if(ok&&state.capture?.source==='live')await autoSelect();return ok;}
async function autoSelect(){const id=state.capture?.id;if(!id)return false;try{state.selecting=true;updateControls();event('SAM is finding every object on the mat…');const selection=await api(`/api/captures/${id}/auto_select`,taskName()==='pickplace'?{previous:state.lastChoice||null}:{});if(state.capture?.id!==id)return false;state.selection=selection;state.plan=null;await loadMasks();
 if(taskName()==='pickplace'){state.selectionRole=selection.from?'to':'from';event(`SAM found ${selection.objects.length} object${selection.objects.length===1?'':'s'}: ${selection.objects.map(objectName).join(', ')}`+(selection.from&&selection.to?` · keeping From ${objectName(selection.objects.find(o=>o.id===selection.from))} → ${pickplaceTarget(selection)}`:' · click From, then To'));return !!(selection.from&&selection.to);}state.selectionRole=selectionRoles().find(r=>!selection[r])||'from';const name=item=>!item?'—':item.kind==='place'?'empty spot':item.kind==='cube'&&item.color?item.color+' cube':item.kind;event(`SAM auto-selected · From ${name(selection.from)} · To ${name(selection.to)} · ${selection.obstacles.length} obstacle${selection.obstacles.length===1?'':'s'}`+(selection.auto?.notes?.length?' · '+selection.auto.notes.join(' '):''));return selectionRoles().every(r=>selection[r]);}catch(e){toast(e.message,true);return false;}finally{state.selecting=false;updateControls();}}
async function operation(name){if(name!=='plan'&&motionMismatch()){toast('Motion settings changed. Generate a new plan and run simulation.',true);return false;}try{state.busy=true;updateControls();if(name==='simulate'&&state.plan){state.plan.outcome='not_simulated';state.plan.transfer_checks_passed=false;delete state.plan.video;setVideo(null);blank('sim','Running contact physics…');}if(name==='plan'){state.plan=null;simImage=null;$('#plan-result').classList.add('hidden');setVideo(null);blank('sim','Computing bounded IK…');}const result=await api(`/api/plans/${state.capture.id}/${name}`,{motion_technique:$('#motion-technique').value,timing_profile:$('#motion-timing').value,speed_scale:Number($('#motion-speed').value)/100,wrist_check:!!$('#wrist-check')?.checked});event({plan:'Solving bounded IK for approach, grasp, lift and release…',simulate:'Running contact-driven MuJoCo simulation and rendering…',execute:'Sending the simulated actions to SO-101.'}[name]);if(name==='execute'){state.plan.executed=true;state.robot.running=true;state.busy=false;}else{$('#step-'+name).classList.add('busy');state.lastJob='';}updateControls();return true;}catch(e){state.busy=false;toast(e.message,true);}updateControls();return false;}
function waitJob(stage){const id=state.capture?.id;return new Promise((resolve,reject)=>{const tick=async()=>{if(!state.runAll)return reject(new Error('Run all stopped'));try{const job=(await api('/api/status')).job;if(job&&job.id===id&&job.stage===stage&&['complete','error'].includes(job.status))return job.status==='error'?reject(new Error(job.error)):resolve(job.plan);}catch{}setTimeout(tick,600);};setTimeout(tick,600);});}
async function returnHome(check){
 // Same Home command as the Return Home button; wait until the arm has stopped at Home.
 const before=state.robot.error;
 for(let attempt=0;attempt<3;attempt++){
  const result=await api('/api/robot/home',{});
  if(result.already_home){event(attempt?'Home reached · capturing the scene':'Already Home · capturing the scene');return;}
  event(result.homing?`Returning Home directly · ${Math.round(result.duration_s)} s`:'Returning Home…');
  const started=Date.now();
  while(true){await new Promise(r=>setTimeout(r,500));check();const robot=(await api('/api/status')).robot;state.robot=robot;updateControls();
   if(robot.error&&robot.error!==before&&!robot.running)throw new Error('Home stopped: '+robot.error);
   if(!robot.running&&Date.now()-started>800)break;
   if(Date.now()-started>60000)throw new Error('Home took too long');}
 }
 throw new Error('The arm did not reach Home; capture was not taken.');}
async function planUntilPass(step,check,recapture){
 // Plan and simulate repeatedly with other speeds and motion techniques until one passes
 // every check. Each try is a full plan + contact simulation; nothing is relaxed.
 const firstTechnique=$('#motion-technique').value,firstSpeed=Number($('#motion-speed').value);
 const variants=[];
 for(const technique of [firstTechnique,...['natural','flow','gentle','classic'].filter(t=>t!==firstTechnique)])
  for(let speed=firstSpeed;speed>=50;speed-=25)variants.push({technique,speed});
 if(!variants.some(v=>v.speed===50&&v.technique==='classic'))variants.push({technique:'classic',speed:50});
 let result=null,lastError='';
 for(const [index,variant] of variants.entries()){
  check();
  // Execution needs a capture under 180 s old.
  if(Date.now()/1000-state.capture.captured_at>130){
   if(!recapture)throw new Error('The Wrist check is too old to execute; return Home and run again.');
   step('Capture is getting old · capturing the scene again');if(!await doCapture())throw new Error('Capture failed');if(!selectionRoles().every(r=>state.selection?.[r]))throw new Error('SAM could not select From and To on the new capture.');}
  $('#motion-technique').value=variant.technique;$('#motion-speed').value=String(variant.speed);updateControls();
  const label=`try ${index+1}/${variants.length} · ${motionLabels[variant.technique]} · ${variant.speed}%`;
  try{
   step(`Planning a route around every obstacle · ${label}`);if(!await operation('plan'))throw new Error('Planning did not start');await showPlan(await waitJob('plan'));check();
   step(`Testing the plan in contact physics · ${label}`);if(!await operation('simulate'))throw new Error('Simulation did not start');result=await waitJob('simulate');await showPlan(result);check();
  }catch(e){if(!state.runAll)throw e;lastError=e.message;result=null;event(`Run all · ${label} failed: ${e.message}`);continue;}
  if(result.outcome==='success'&&result.transfer_checks_passed)break;
  lastError=result.outcome!=='success'?'Outcome: '+result.outcome:(result.transfer_check_failures||[]).join('; ');
  event(`Run all · ${label} did not pass: ${lastError} · trying another plan`);
  // Going slower only helps a joint that trails its command; otherwise change technique.
  const failures=result.transfer_check_failures||[];
  if(!(result.outcome==='success'&&failures.length&&failures.every(f=>f.includes('lower Travel speed')))){
   const next=variants.findIndex((v,k)=>k>index&&v.technique!==variant.technique);
   if(next>index+1)variants.splice(index+1,next-index-1);}
 }
 if(!result)throw new Error('No plan could be simulated; the robot was not moved. Last error: '+lastError);
 if(result.outcome!=='success'||!result.transfer_checks_passed){const failures=result.transfer_check_failures||[];const reach=failures.length&&failures.every(f=>f.includes('trails its command'));
  throw new Error('Simulation did not pass; the robot was not moved. '+(reach?'Even at 50% travel speed a joint trailed its command: the object or its destination is probably too far from the robot. Move it closer and try again. ':'')+(failures.join('; ')||'Outcome: '+result.outcome));}
 if(!state.robot.connected||Date.now()/1000-result.captured_at>180)throw new Error('Robot disconnected or capture expired; the robot was not moved.');
 return result;}
async function waitExecution(check){
 // Until the hardware run of the current capture has finished and been recorded.
 const id=state.capture?.id;
 for(let tries=0;tries<400;tries++){await new Promise(r=>setTimeout(r,500));check();const status=await api('/api/status');const job=status.job;
  if(!status.robot.running&&job?.id===id&&job.stage==='execute'&&['complete','error'].includes(job.status)){if(job.status==='error')throw new Error(job.error);return job.plan;}}
 throw new Error('The arm did not finish in time');}
async function lookAndGrasp(step,check){
 // The arm is still above the object: measure it with the Wrist camera, then plan,
 // simulate and execute the rest from the measured pose.
 const parent=state.capture.id;await waitExecution(check);check();
 step('Wrist camera check · segmenting the object right above it');
 const result=await api(`/api/plans/${parent}/refine`,{});const r=result.refinement;
 event(`Wrist check · object ${(r.shift_m*1000).toFixed(1)} mm${r.turn_rad?` and ${(Math.abs(r.turn_rad)*180/Math.PI).toFixed(1)}° turn`:''} from the top-camera estimate · fit ${r.rmse_px.toFixed(1)} px`);
 state.request++;state.capture=result.capture;state.activeCapture=result.capture.id;state.plan=null;history.replaceState(null,'','?run='+result.capture.id);await capturedImage().catch(()=>{});updateControls();
 await planUntilPass(t=>step('Wrist-checked grasp · '+t),check,false);
 step('Executing the Wrist-checked grasp and place · Esc or Stop motion halts it');if(!await operation('execute'))throw new Error('Execution was refused by the safety checks');}
async function runAll(){
 if(state.runAll||state.busy||state.selecting||state.robot.running)return;state.runAll=true;updateControls();
 const step=text=>{$('#run-all-status').textContent=text;event('Run all · '+text);},check=()=>{if(!state.runAll)throw new Error('Run all stopped');};
 try{
  // Reuse a fresh, unexecuted capture whose selection the user may have corrected; otherwise capture now.
  const reuse=state.page==='new'&&state.capture?.source==='live'&&state.capture.id===state.activeCapture&&!state.plan?.executed&&Date.now()/1000-state.capture.captured_at<90&&selectionRoles().every(r=>state.selection?.[r]);
  if(!reuse){step('1/5 Returning Home so the camera sees the whole mat');await returnHome(check);check();step('2/5 Capturing scene and selecting objects with SAM');if(!await doCapture())throw new Error('Capture failed');}
  check();if(!selectionRoles().every(r=>state.selection?.[r]))throw new Error('Choose From and To in the camera image, then press Run all again. Run all keeps your choice on later captures.');
  const result=await planUntilPass(t=>step('3–4/5 '+t),check,true);
  const look=result.look_index!==undefined&&result.look_index!==null;
  step(look?'5/5 Moving above the object for the Wrist camera check · Esc or Stop motion halts it':'5/5 Executing on SO-101 · Esc or Stop motion halts it');if(!await operation('execute'))throw new Error('Execution was refused by the safety checks');
  if(look)await lookAndGrasp(step,check);
  step('Running on the robot · same actions and timing as the simulation');
 }catch(e){if(state.runAll||e.message!=='Run all stopped')toast(e.message,true);$('#run-all-status').textContent='Run all stopped: '+e.message;}
 finally{state.runAll=false;updateControls();}}
$('#run-all').onclick=runAll;
async function calibrateCamera(){
 // Move the arm to the pose of the dataset reference frame, then compare the live top
 // image with that frame (arm included). Every capture is also lined up with the
 // dataset view automatically when the camera has moved.
 if(state.runAll||state.busy||state.robot.running)return;state.runAll=true;updateControls();
 const say=t=>{$('#run-all-status').textContent=t;event(t);};
 try{say('Camera check · moving the arm to the dataset frame pose · Esc or Stop motion halts it');
  let r=null;
  for(let attempt=0;attempt<4&&!r;attempt++){
   await api('/api/camera/pose',{correct:attempt>0});const started=Date.now();
   while(true){await new Promise(r=>setTimeout(r,500));if(!state.runAll)throw new Error('Stopped');const robot=(await api('/api/status')).robot;state.robot=robot;updateControls();
    if(robot.error&&!robot.running)throw new Error('Move stopped: '+robot.error);if(!robot.running&&Date.now()-started>800)break;if(Date.now()-started>30000)throw new Error('The move took too long');}
   await new Promise(r=>setTimeout(r,600));
   say('Camera check · comparing the top camera with the dataset frame');
   try{r=await api('/api/camera/check',{});}catch(e){if(!/from the dataset frame pose/.test(e.message)||attempt===3)throw e;say('Camera check · '+e.message+' · correcting the pose');}}
  const text=r.moved_px==null?'Camera check: '+(r.note||'could not compare with the dataset frame'):r.moved_px<=1?`Camera matches dataset episode ${r.episode} (${r.moved_px.toFixed(1)} px, arm pose within ${r.pose_error_deg.toFixed(1)}°)`:r.moved_px<=25?`Camera moved ${r.moved_px.toFixed(1)} px (turn ${r.turn_deg.toFixed(2)}°) from dataset episode ${r.episode} · captures are lined up with it automatically`:`Camera moved ${r.moved_px.toFixed(0)} px from the dataset frame · put it back where it was`;
  say(text);}
 catch(e){if(e.message!=='Stopped')toast(e.message,true);$('#run-all-status').textContent='Camera check stopped: '+e.message;}
 finally{state.runAll=false;updateControls();}}
$('#calibrate-camera').onclick=calibrateCamera;
async function poll(){if(state.restarting)return;try{const status=await api('/api/status');state.robot=status.robot;state.activeCapture=status.current;if(state.mode==='live'&&!state.capture)$('#real-label').textContent='LIVE CAMERA';if(state.robot.running&&state.mode==='live'&&state.capture?.id===state.robot.execution&&state.plan?.video){$('#real-label').textContent='LIVE ROBOT';video.playbackRate=1;$('#speed').value='1';if(Math.abs(video.currentTime-state.robot.elapsed)>.2)video.currentTime=state.robot.elapsed;if(video.paused&&playback.ready)video.play().catch(()=>{});playbackLabel('Live robot / sim clock');}$('#connection').className='status'+(state.robot.connected?' connected':'');$('#connection').innerHTML=`<i></i>${state.robot.running?'Executing motion':state.robot.connected?(state.robot.torque_enabled===false?'Torque off · arm free':state.robot.torque_enabled===true?'SO-101 connected':'Torque state unknown'):'Robot offline'}`;const job=status.job;if(job?.status==='running'||job?.status==='recording'){state.busy=true;$('#step-'+job.stage).classList.add('busy');}else if(job&&['complete','error'].includes(job.status)){const key=job.id+job.stage+job.status;if(state.lastJob!==key){state.lastJob=key;refreshRuns().catch(()=>{});state.busy=false;$$('.step.busy').forEach(e=>e.classList.remove('busy'));if(job.status==='error'){if(state.page==='new')toast(job.error,true);}else if(job.stage==='prepare'&&state.page==='new'){state.busy=false;event(job.plan.execution?.completed?'Home reached · camera view clear · capture a new scene':'Home stopped · '+(job.plan.execution?.error||'motion interrupted'));}else if(state.capture?.id===job.id){await showPlan(job.plan);event(job.stage==='execute'?`Hardware ${job.plan.execution?.completed?'completed':'stopped'} · synchronized real/sim recording ready`:job.stage==='plan'?`IK plan ready · ${job.plan.frames} actions`:`Physics result: ${pretty(job.plan.outcome)} · ${job.plan.transfer_checks_passed?'transfer checks passed':'transfer checks blocked execution'}`);}}}if(state.robot.error&&state.robot.error!==state.robot.lastError&&state.robot.error!==state.lastRobotError){state.lastRobotError=state.robot.error;toast(state.robot.error,true);}updateControls();}catch(e){$('#connection').innerHTML='<i></i>Server unavailable';state.robot={connected:false};updateControls();}setTimeout(poll,1000);}
async function livePreview(){if(state.restarting)return;if(state.page==='new'&&state.mode==='live'&&(!state.capture||(state.capture.id===state.activeCapture&&(state.robot.running||(state.plan?.executed&&!state.plan?.execution?.video))))&&state.robot.connected&&!liveBusy){liveBusy=true;try{const view=state.view,id=state.capture?.id;const frame=await image('/api/live/'+view+'?t='+Date.now());if(state.page==='new'&&state.mode==='live'&&state.view===view&&state.capture?.id===id&&(!state.capture||(state.capture.id===state.activeCapture&&(state.robot.running||(state.plan?.executed&&!state.plan?.execution?.video))))){realImage=frame;draw();}}catch{}finally{liveBusy=false;}}setTimeout(livePreview,150);}
function runTitle(run){const date=new Date(run.captured_at*1000).toLocaleString('en-GB',{day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'});const kind=run.purpose==='home'?'Home':run.task==='stack'?'Cube stack':run.task==='tape'?'Tape in bowl':run.source==='live'?'Live pick/place':'Rehearsal';return `${date} · ${kind}${run.motion_technique?' · '+(motionLabels[run.motion_technique]||run.motion_technique)+' '+Math.round((run.speed_scale??1)*100)+'%':''} · ${run.id.slice(0,6)}`;}
function runResult(run){if(run.status==='executing')return 'Running on robot';if(run.status==='recording')return 'Preparing recording';if(run.error)return 'Interrupted';if(run.hardware_completed===true)return run.physical_outcome==='success'?'Placement verified':run.physical_outcome==='failed'?'Commands complete · placement failed':'Commands complete';if(run.hardware_completed===false)return 'Stopped early';if(run.simulation_outcome==='success')return 'Simulation passed · not executed';return pretty(run.status);}
function historySidebar(){applyLayout();}
function renderRuns(){const query=$('#search').value.toLowerCase();$('#episodes').replaceChildren();for(const run of state.runs){const title=runTitle(run),result=runResult(run);if(query&&!`${title} ${result} ${run.id}`.toLowerCase().includes(query))continue;const b=document.createElement('button');b.className='episode-item run-item'+(state.capture?.id===run.id?' active':'');const name=document.createElement('strong'),detail=document.createElement('small');name.textContent=title;detail.textContent=result;b.append(name,detail);b.disabled=state.busy||state.selecting||state.robot.running;b.onclick=()=>loadRun(run.id);$('#episodes').append(b);}}
async function refreshRuns(){const data=await api('/api/runs');state.runs=data.runs;state.activeCapture=data.active_capture;$('#run-count').textContent=data.runs.length;const select=$('#run-select');select.replaceChildren(new Option('Select a recorded run…',''));for(const run of data.runs)select.append(new Option(`${runTitle(run)} · ${runResult(run)}`,run.id));select.value=state.capture?.id||'';if(state.historyMode)renderRuns();applyLayout();}
async function loadRun(id){if(!id||state.busy||state.selecting||state.robot.running)return;try{let request=++state.request;video.pause();const saved=await api('/api/plans/'+id);if(request!==state.request)return;state.capture=saved;state.plan=null;state.selection=saved.selection||null;state.historyMode=true;state.page='history';realImage=null;simImage=null;masks={};$('#source').value=saved.source==='live'?'live':'dataset';history.replaceState(null,'','?run='+id);historySidebar();setMode('live');request=state.request;$('#plan-result').classList.add('hidden');if(saved.frames)await showPlan(saved);if(request!==state.request||state.capture?.id!==id)return;await loadMasks();if(request!==state.request||state.capture?.id!==id)return;await capturedImage();if(request!==state.request||state.capture?.id!==id)return;const run=state.runs.find(r=>r.id===id);$('#run-summary').textContent=run?`${runTitle(run)} · ${runResult(run)}${run.error?' · '+run.error:''} · Playback only · no robot commands`:'Saved run · Playback only · no robot commands';$('#run-summary').classList.remove('hidden');$('#run-select').value=id;renderRuns();updateControls();event('Opened saved run '+id.slice(0,8));}catch(e){toast(e.message,true);}}
$('#nav-history').onclick=showHistory;
$('#run-select').onchange=()=>loadRun($('#run-select').value);$('#refresh-runs').onclick=()=>refreshRuns().catch(e=>toast(e.message,true));
function selectionRoles(){return ['from','to'];}
const objectName=o=>!o?'—':(o.color?o.color+' ':'')+(o.kind==='tape'?'tape roll':o.kind);
function pickplaceTarget(selection,short=false){const to=selection?.to;if(!to)return null;if(to.area)return short?'empty spot':'to an empty spot';const o=selection.objects.find(x=>x.id===to.object);return o?(short?objectName(o):o.kind==='bowl'?'into the bowl':'onto the '+objectName(o)):null;}
function pickplaceDraw(r){
 const sel=state.selection,done=new Set(),byId=Object.fromEntries(sel.objects.map(o=>[o.id,o]));
 for(const o of sel.objects)if(!done.has(o.mask_file)&&masks.objects?.[o.mask_file]){r.drawImage(masks.objects[o.mask_file],0,0,640,480);done.add(o.mask_file);}
 sel.objects.forEach(o=>{
  const role=o.id===sel.from?'FROM':sel.to?.object===o.id?'TO':'',color=role==='FROM'?'#e0662a':role==='TO'?'#2f9e57':'#ffffff';
  const below=byId[o.support],shared=below&&below.mask_file===o.mask_file;let [x,y]=o.point;if(shared)y-=12;
  r.font=role?'bold 15px sans-serif':'bold 12px sans-serif';r.lineWidth=3;r.strokeStyle='rgba(0,0,0,.55)';const text=(role?role+' · ':'')+objectName(o)+(below?(below.kind==='bowl'?' (in bowl)':' (on '+objectName(below)+')'):'');
  const at=[Math.max(4,Math.min(636-r.measureText(text).width,x-r.measureText(text).width/2)),Math.max(16,y-12)];r.strokeText(text,...at);r.fillStyle=color;r.fillText(text,...at);
  r.strokeStyle=color;r.lineWidth=role?3:1.5;r.beginPath();r.arc(x,y,role?7:4,0,Math.PI*2);r.stroke();});
 if(sel.to?.area&&sel.to.point){const [x,y]=sel.to.point;r.strokeStyle='#2f9e57';r.lineWidth=3;r.beginPath();r.arc(x,y,13,0,Math.PI*2);r.stroke();r.font='bold 15px sans-serif';r.lineWidth=3;r.strokeStyle='rgba(0,0,0,.55)';r.strokeText('TO · EMPTY SPOT',x-62,Math.max(16,y-20));r.fillStyle='#2f9e57';r.fillText('TO · EMPTY SPOT',x-62,Math.max(16,y-20));}
}
function rememberChoice(){const sel=state.selection;if(!sel?.objects||!sel.from||!sel.to)return;const pick=id=>{const o=sel.objects.find(x=>x.id===id);return o&&{kind:o.kind,color:o.color??null,position:o.position};};state.lastChoice={from:pick(sel.from),to:sel.to.object?{object:pick(sel.to.object)}:{area:sel.to.area,point:sel.to.point}};}
function taskName(){return state.capture?.task??$('#task').value;}
function updateSelection(){
 const visible=state.page==='new'&&state.mode==='live'&&$('#source').value==='live';$('#sam-selector').classList.toggle('hidden',!visible);
 if(taskName()==='pickplace'){
  const enabled=visible&&state.capture?.id===state.activeCapture&&state.capture?.source==='live'&&state.view==='top'&&!state.busy&&!state.selecting&&!state.robot.running&&!state.plan?.executed&&!state.runAll;
  real.classList.toggle('selectable',!!enabled);const sel=state.selection,objects=sel?.objects||[],from=objects.find(o=>o.id===sel?.from),target=pickplaceTarget(sel);
  for(const role of ['from','to','obstacle']){const b=$('#select-'+role);b.classList.toggle('hidden',role==='obstacle');b.disabled=!enabled;b.classList.toggle('selected',state.selectionRole===role);b.setAttribute('aria-pressed',String(state.selectionRole===role));}
  $('#select-from').textContent='From · '+(from?objectName(from)+' ✓':'Object to pick');$('#select-to').textContent='To · '+(target?pickplaceTarget(sel,true)+' ✓':'Object or empty spot');
  const found=objects.map(o=>objectName(o)+(o.support!=='mat'?(objects.find(b=>b.id===o.support)?.kind==='bowl'?' in bowl':' on '+objectName(objects.find(b=>b.id===o.support))):'')).join(', ');
  $('#sam-status').textContent=state.selecting?'Segmenting with SAM… finding every object on the mat.':!state.capture?'Capture a scene; SAM finds every object, then click From and To.':state.plan?.executed?'Executed · capture a new scene to choose again.':state.view!=='top'?'Switch to Top camera to choose From and To.':!objects.length?'No known objects found.':`Found: ${found}. `+(from&&target?`Move the ${objectName(from)} ${target}. Every other object is simulated too. Click to change.`:state.selectionRole==='from'?'Click the object to pick (From).':'Click where it goes (To): another object, or an empty spot on the mat.');
  return;}
 const enabled=visible&&state.capture?.id===state.activeCapture&&state.capture?.source==='live'&&state.view==='top'&&!state.busy&&!state.selecting&&!state.robot.running&&!state.plan?.executed&&!state.runAll;
 real.classList.toggle('selectable',!!enabled);const roles=selectionRoles(),task=taskName(),obstacles=state.selection?.obstacles||[];
 const labels={from:task==='tape'?'From · Tape to pick':task==='unstack'?'From · Top cube':'From · Cube to pick',to:task==='stack'?'To · Cube to stack on':task==='unstack'?'To · Empty spot':'To · Bowl',obstacle:`+ Obstacle${obstacles.length?` (${obstacles.length})`:''}`};
 for(const role of ['from','to','obstacle']){const b=$('#select-'+role);b.classList.remove('hidden');b.disabled=!enabled;b.classList.toggle('selected',state.selectionRole===role);b.setAttribute('aria-pressed',String(state.selectionRole===role));b.textContent=labels[role]+(role!=='obstacle'&&state.selection?.[role]?' ✓':'');}
 const ready=roles.every(role=>state.selection?.[role]),kinds=obstacles.map(o=>o.kind==='cube'&&o.color?o.color+' cube':o.kind==='tape'?'tape roll':o.kind).join(', ');
 const target={from:task==='tape'?'the tape roll to pick':task==='unstack'?'the top cube of the stack':'the cube to pick',to:task==='stack'?'the destination cube':task==='unstack'?'an empty spot on the mat':'the bowl',obstacle:'any cube, tape roll or bowl SAM missed'}[state.selectionRole];
 $('#sam-status').textContent=state.selecting?'Segmenting with SAM… finding every object on the mat.':!state.capture?'Choose a task and capture a scene; SAM then selects every object automatically.':state.plan?.executed?'Executed · capture a new scene to select objects again.':state.view!=='top'?'Switch to Top camera to change the selection.':(ready?`Selection ready · From ${(state.selection.from.score*100).toFixed(0)}% · To ${state.selection.to.kind==='place'?'empty spot':(state.selection.to.score*100).toFixed(0)+'%'} · Obstacles: ${kinds||'none'} · Plans always avoid obstacles. Click to change: ${target}.`:`Click ${target} in the left camera image.`)+(state.selection?.auto?.notes?.length?' '+state.selection.auto.notes.join(' '):'');
}
async function loadMasks(){const id=state.capture?.id,selection=state.selection,next={obstacles:[],objects:{}};
 if(selection?.objects&&taskName()==='pickplace'){for(const o of selection.objects)if(!next.objects[o.mask_file])next.objects[o.mask_file]=await image(`/media/studio/${id}/${o.overlay_file}`);if(state.capture?.id===id&&state.selection===selection)masks=next;draw();return;}for(const role of ['from','to']){if(selection?.[role]?.overlay_file)next[role]=await image(`/media/studio/${id}/${selection[role].overlay_file}`);}for(const item of selection?.obstacles||(selection?.obstacle?[selection.obstacle]:[]))next.obstacles.push(await image(`/media/studio/${id}/${item.overlay_file}`));if(state.capture?.id===id&&state.selection===selection)masks=next;draw();}
for(const role of ['from','to','obstacle'])$('#select-'+role).onclick=()=>{state.selectionRole=role;updateSelection();};
real.onclick=async e=>{
 if(state.capture?.id!==state.activeCapture||state.mode!=='live'||state.view!=='top'||state.busy||state.selecting||state.robot.running||!state.capture||state.plan?.executed||!real.classList.contains('selectable'))return;
 const box=real.getBoundingClientRect(),point=[(e.clientX-box.left)*640/box.width,(e.clientY-box.top)*480/box.height],id=state.capture.id,role=state.selectionRole;
 try{state.selecting=true;updateControls();const selection=await api(`/api/captures/${id}/select`,{role,point});if(state.capture?.id!==id)return;state.selection=selection;state.plan=null;simImage=null;$('#plan-result').classList.add('hidden');if(state.mode==='live'){setVideo(null);blank('sim','Objects selected · generate a new plan');setOutcome('objects selected');}await loadMasks();if(taskName()==='pickplace'){rememberChoice();state.selectionRole=role==='from'?'to':'to';event(role==='from'?`From · ${objectName(selection.objects.find(o=>o.id===selection.from))}`:`To · ${pickplaceTarget(selection)}`);}else{state.selectionRole=role==='obstacle'?'obstacle':selectionRoles().find(r=>!selection[r])||role;event(role==='obstacle'?`Obstacle added · ${selection.obstacles.length} obstacles`:selection[role].kind==='place'?'To set · empty mat spot checked clear of every object':`SAM selected ${role.toUpperCase()} · ${(selection[role].score*100).toFixed(1)}% confidence`);}}catch(e){toast(e.message,true);}finally{state.selecting=false;updateControls();}
};
$('#episode-select').onchange=()=>loadEpisode(Number($('#episode-select').value));
$('#nav-data').onclick=showDatasets;$('#nav-live').onclick=newRun;$('#back-datasets').onclick=showDatasets;$('#search').oninput=renderList;$('#filter').onclick=()=>{state.filter=state.filter==='all'?'failed':'all';$('#filter').textContent=state.filter==='all'?'All results ▾':'Unresolved ▾';renderList();};$('#source').onchange=sourceChanged;$('#task').onchange=async()=>{await newRun();sourceChanged();};$('#joint').onchange=chart;$('#sim-appearance').value=state.appearance;$('#sim-appearance').onchange=changeAppearance;
$$('[data-view]').forEach(b=>b.onclick=async()=>{state.view=b.dataset.view;updateSelection();$$('[data-view]').forEach(x=>x.classList.toggle('selected',x===b));$$('.view-name').forEach(x=>x.textContent=state.view.toUpperCase());if(state.mode==='live'&&!state.plan?.execution?.video){realImage=null;blank('real','Loading camera…');if(state.view==='wrist'&&!state.plan?.video)blank('sim','Wrist view available after simulation.');draw();if(!state.robot.running){try{await capturedImage();}catch(e){toast(e.message,true);}}}draw();});
$('#play').onclick=()=>playback.toggle();$('#speed').onchange=()=>video.playbackRate=Number($('#speed').value);
$('#timeline').oninput=()=>{const f=Number($('#timeline').value);if(video.getAttribute('src')){video.currentTime=f/30;state.frame=f;updateTimeline();}else if(state.mode==='dataset'){state.frame=f;updateTimeline();clearTimeout(seekTimer);const index=state.index;seekTimer=setTimeout(async()=>{try{const im=await image(`/api/episodes/${index}/frame/${f}${appearanceQuery()}`);if(state.mode==='dataset'&&state.index===index&&state.frame===f){combined=im;draw();}}catch(e){toast(e.message,true);}},200);}};
async function prepareArm(command){try{state.busy=true;updateControls();const result=await api('/api/robot/'+command,{});state.capture=null;$('#run-summary').classList.add('hidden');history.replaceState(null,'','/?page=new');state.selection=null;masks={};state.plan=null;realImage=null;simImage=null;state.lastJob='';$('#plan-result').classList.add('hidden');$('#source').value='live';setMode('live');sourceChanged();if(result.already_home){state.busy=false;event('Already Home · capture a new scene');}else if(command==='home'){state.busy=false;event(result.homing?`Returning Home directly · ${Math.round(result.duration_s)} s · no camera or collision checks`:'Returning Home…');}else{state.robot=result;state.busy=false;event(command==='torque_off'?'Torque off · support and position the arm by hand':'Torque on · holding current pose · capture a new scene');}}catch(e){state.busy=false;toast(e.message,true);}updateControls();}
$('#home').onclick=()=>prepareArm('home');$('#torque-off').onclick=()=>prepareArm('torque_off');$('#torque-on').onclick=()=>prepareArm('torque_on');
$('#motion-speed').oninput=$('#motion-technique').onchange=$('#motion-timing').onchange=()=>{updateControls();if(motionMismatch()){setVideo(null);blank('sim','Motion settings changed · generate a new plan');setOutcome('plan needs update');}};
$('#capture').onclick=doCapture;$('#plan').onclick=()=>operation('plan');$('#simulate').onclick=()=>operation('simulate');$('#execute').onclick=async()=>{const look=state.plan?.look_index!==undefined&&state.plan?.look_index!==null;if(!await operation('execute')||!look)return;
 // Execute also finishes a Wrist-checked grasp: stop above the object, measure, re-plan, simulate, run.
 state.runAll=true;updateControls();const step=text=>{$('#run-all-status').textContent=text;event(text);},check=()=>{if(!state.runAll)throw new Error('Stopped');};
 try{await lookAndGrasp(step,check);step('Running on the robot · same actions and timing as the simulation');}catch(e){if(e.message!=='Stopped')toast(e.message,true);$('#run-all-status').textContent='Stopped: '+e.message;}finally{state.runAll=false;updateControls();}};$('#connect').onclick=async()=>{try{$('#connect').disabled=true;const command=state.robot.connected?'disconnect':'connect';event(command==='connect'?'Connecting SO-101 and camera streams…':'Releasing hardware connection with torque retained…');state.robot=await api('/api/robot/'+command,{});toast(command==='connect'?'SO-101 connected. Ready to capture.':'Robot disconnected; servo torque retained.');}catch(e){toast(e.message,true);}finally{updateControls();}};
async function stop(){if(state.runAll){state.runAll=false;$('#run-all-status').textContent='Run all stopped.';}try{await api('/api/robot/stop',{});toast('Motion stopped. Holding measured pose.');event('Software stop requested · future commands cancelled.');}catch(e){toast(e.message,true);}}$('#stop').onclick=stop;document.addEventListener('keydown',e=>{if(e.key==='Escape')stop();});

function applyLayout(){
 const dataset=state.page==='datasets',selected=dataset&&!!state.selectedDataset,historyPage=state.page==='history',fresh=state.page==='new',generate=state.page==='generate',objects=state.page==='objects';
 document.body.dataset.page=state.page;
 state.historyMode=historyPage;
 for(const [id,on] of [['nav-data',dataset],['nav-live',fresh],['nav-history',historyPage],['nav-generate',generate],['nav-objects',objects]])$('#'+id).classList.toggle('active',on);
 $('#page-title').textContent=dataset?(selected?state.selectedDataset:'Real Datasets'):fresh?'New Run':generate?'Generate data':objects?'3D-Objects':'Run history';
 $('#page-description').textContent=dataset?(selected?'Browse episodes and compare real and simulated motion.':'Choose a dataset to explore its recorded episodes.'):fresh?'Start a fresh session from the current robot and camera state.':generate?'Create, inspect, and export synthetic demonstrations for policy research.':objects?'Explore the calibrated object models and see how each task uses them.':'Review actual robot executions, including interrupted and unsuccessful runs.';
 $('#generation-page').classList.toggle('hidden',!generate);
 $('#objects-page').classList.toggle('hidden',!objects);
 $('#replay-launch').classList.toggle('hidden',!selected||importedDataset());
 if(!selected)replayBuilder.suspend();
 if(!objects)objectsView.suspend();
 if(!generate)generationViewer.suspend();
 $('#dataset-browser').classList.toggle('hidden',!dataset||selected);
 $('#import-page').classList.toggle('hidden',!dataset||selected);
 if(!dataset||selected)importView.suspend();
 $('.dataset-block').classList.toggle('hidden',!selected);
 $('.episode-heading').classList.toggle('hidden',!selected&&!historyPage);
 $('.search').classList.toggle('hidden',!selected&&!historyPage);
 $('#episodes').classList.toggle('hidden',!selected&&!historyPage);
 $('#new-run-note').classList.toggle('hidden',!fresh);
 $('#list-heading').textContent=historyPage?'RECORDED RUNS':'EPISODES';
 $('#search').placeholder=historyPage?'Search date, ID, or Home…':'Find episode…';
 $('#filter').classList.toggle('hidden',!selected);
 $('.summary').classList.toggle('hidden',!selected);
 $('.profile-tag').classList.toggle('hidden',!selected);
 $('.workbench-grid').classList.toggle('hidden',objects||generate||dataset&&!selected||historyPage&&!state.runs.length);
 $('.workbench-grid').classList.toggle('viewer-only',!fresh);
 $('.pipeline').classList.toggle('hidden',!fresh);
 $('.history-picker').classList.toggle('hidden',!historyPage);
 $('#history-empty').classList.toggle('hidden',!historyPage||!!state.runs.length);
 $('#episode-select').classList.toggle('hidden',!selected);
 $('.chart-panel').classList.toggle('hidden',!selected);
 $('.activity').classList.toggle('hidden',!fresh);
 $('#sam-selector').classList.toggle('hidden',!fresh);
 if(!historyPage)$('#run-summary').classList.add('hidden');
}
function resetViewer(){generationViewer.suspend();state.request++;video.pause();setVideo(null);state.episode=null;state.capture=null;state.plan=null;state.selection=null;state.selectionRole='from';state.frame=0;state.view='top';video.playbackRate=1;$('#speed').value='1';combined=null;realImage=null;simImage=null;masks={};$('#plan-result').classList.add('hidden');$$('[data-view]').forEach(b=>b.classList.toggle('selected',b.dataset.view==='top'));$$('.view-name').forEach(e=>e.textContent='TOP');real.getContext('2d').clearRect(0,0,640,480);sim.getContext('2d').clearRect(0,0,640,480);$('#search').value='';}
function showDatasets(){if(state.busy||state.selecting||state.robot.running)return;resetViewer();state.page='datasets';state.selectedDataset=null;state.mode='dataset';history.replaceState(null,'','/');renderDatasets();applyLayout();updateControls();importView.show();refreshDatasets();}
async function refreshDatasets(){try{const catalog=await api('/api/datasets');state.datasets=catalog.datasets;$('#dataset-count').textContent=state.datasets.length;if(state.page==='datasets'&&!state.selectedDataset)renderDatasets();}catch{}}
function renderDatasets(){const cards=$('#dataset-cards');cards.replaceChildren();for(const dataset of state.datasets){const button=document.createElement('button');button.className='dataset-card';const icon=document.createElement('span');icon.className='dataset-icon';icon.textContent='▥';const title=document.createElement('strong');title.textContent=dataset.name;const detail=document.createElement('span');detail.textContent=`${dataset.episode_count} episodes · ${dataset.fps} Hz`+(dataset.source==='import'?' · imported, auto-calibrated':'');const open=document.createElement('b');open.textContent='Explore episodes →';button.append(icon,title,detail,open);button.onclick=()=>chooseDataset(dataset.id);cards.append(button);}}
async function chooseDataset(id){try{state.busy=true;updateControls();const request=++state.request;const data=await api('/api/datasets/'+encodeURIComponent(id));if(request!==state.request)return;state.page='datasets';state.selectedDataset=id;state.episodes=data.episodes;$('#selected-dataset-name').textContent=id;$('#dataset-meta').textContent=`${data.total} episodes · 30 Hz`;$('#success-rate').innerHTML=`${(data.successes/data.total*100).toFixed(1)}<em>%</em>`;$('#success-total').textContent=`${data.successes} / ${data.total} episodes · task criteria`;$('#episode-select').replaceChildren();for(const e of data.episodes)$('#episode-select').append(new Option(`Episode ${String(e.episode).padStart(2,'0')} · ${e.outcome==='success'?'success':'unresolved'}`,e.episode));await loadEpisode(data.episodes[0].episode);}catch(e){toast(e.message,true);}finally{state.busy=false;applyLayout();updateControls();}}
async function newRun(){if(state.busy||state.selecting||state.robot.running)return;try{state.busy=true;updateControls();await api('/api/session/new',{});resetViewer();state.activeCapture=null;state.lastJob='';state.lastRobotError=null;state.page='new';state.historyMode=false;$('#events').replaceChildren();$('#source').value='live';history.replaceState(null,'','/?page=new');setMode('live');sourceChanged();event('New session ready · connect the robot, then capture the current scene.');}catch(e){toast(e.message,true);}finally{state.busy=false;updateControls();}}
async function showHistory(){if(state.busy||state.selecting||state.robot.running)return;resetViewer();state.page='history';state.historyMode=true;state.mode='live';const request=state.request;history.replaceState(null,'','/?page=history');try{await refreshRuns();if(request!==state.request||state.page!=='history')return;if(state.runs.length)await loadRun(state.runs[0].id);else applyLayout();}catch(e){toast(e.message,true);}}
async function restartServer(){if(state.restarting)return;try{await api('/api/server/restart',{});state.restarting=true;state.request++;video.pause();$('#restart-overlay').classList.remove('hidden');updateControls();const previous=state.instance;for(let attempt=0;attempt<60;attempt++){await new Promise(r=>setTimeout(r,500));try{const data=await api('/api/bootstrap');if(data.instance!==previous){location.replace('/?page=new');return;}}catch{}}$('#restart-message').textContent='Server is taking longer than expected. Reload this page to reconnect.';}catch(e){toast(e.message,true);}}
$('#restart-server').onclick=restartServer;

const objectsView=new ObjectsView($('#objects-page'),api);
const importView=new ImportView($('#import-page'),api,toast,async id=>{await refreshDatasets();await chooseDataset(id);});
const replayBuilder=new ReplayDatasetBuilder($('#replay-builder'),api,{onBuilt:job=>showGeneration(job.id,'visualize'),onClose:()=>replayBuilder.suspend()});
async function showObjects(){if(state.busy||state.selecting||state.robot.running)return;resetViewer();state.page='objects';window.scrollTo(0,0);history.replaceState(null,'','/?page=objects');applyLayout();updateControls();await objectsView.load();}
$('#nav-objects').onclick=()=>showObjects().catch(error=>toast(error.message,true));
$('#open-replay-builder').onclick=()=>{video.pause();replayBuilder.open(state.selectedDataset,state.index);};

// Generation has its own job state; opening it never starts a robot session.
const generationViewer=new GenerationVisualizer(api);
const generationCoverage=new GenerationCoverage(api,$('#generation-coverage'),selectGenerationEpisode,generationTaskName);
const generation={tab:'build',jobs:[],job:null,busy:false,submitting:false,cancelling:false,request:0,timer:null,episodeIndex:0,episodesSignature:'',loaded:false,parent:null,deleting:false,deleteId:null,backgrounds:[{id:'calibrated-real',name:'Calibrated real workcell',mode:'calibrated'}],uploading:false,assetsLoaded:false,preview:{request:0,busy:false,timer:null,signature:null}};
const generationZones=new ObjectZonePicker($('#generation-object-zones'),()=>{invalidateGenerationStills();updateGenerationForm();});
const generationActive=job=>['queued','running'].includes(job?.status);
const generationTaskInstructions={green_cube_to_green_bowl:'Put the green cube in the green bowl.',red_cube_to_green_bowl:'Put the red cube in the green bowl.',red_cube_to_green_cube:'Stack the red cube on the green cube.',green_cube_to_red_cube:'Stack the green cube on the red cube.',tape_to_green_bowl:'Put the tape in the green bowl.'};
function generationTaskName(id,options=generation.job?.options){
 if(id==='unstack'){const top=options?.unstack_top==='green'?'green':'red',bottom=top==='red'?'green':'red';return `Lift the ${top} cube off the ${bottom} cube and place it on the empty mat.`;}
 return generationTaskInstructions[id]||pretty(id);
}
const generationTimingName=options=>options?.timing_profile==='recorded'?'Recorded robot pace':'Conservative studio pace';
// One field map keeps restored settings, previews and generation submissions aligned.
const generationFields={
 episodes_per_task:['count',1,'number'],seed:['seed',42,'number'],motion_technique:['motion','natural'],timing_profile:['timing','recorded'],speed_scale:['speed',1,'number'],
 layout_strategy:['layout-strategy','uniform'],randomize_layout:['layout',true,'boolean'],real_background:['real-background',false,'boolean'],real_image_processing:['real-processing',false,'boolean'],visual_randomization:['visual',true,'boolean'],camera_jitter:['camera',false,'boolean'],sensor_augmentation:['sensor',true,'boolean'],physics_randomization:['physics',false,'boolean'],validation_fraction:['validation',.2,'number'],unstack_top:['unstack','red'],
 background_id:['background','calibrated-real'],lighting_randomization:['lighting',false,'boolean'],light_intensity:['light-intensity',1,'number'],light_direction:['light-direction',-30,'number'],light_elevation:['light-elevation',55,'number'],light_spread:['light-spread',45,'number'],shadow_strength:['shadow-strength',.5,'number']
};
function collectGenerationSettings(){
 const settings={name:$('#generation-name').value.trim(),tasks:[...$$('[name="generation-task"]:checked')].map(input=>input.value)};
 for(const [key,[id,fallback,type]] of Object.entries(generationFields)){const input=$('#generation-'+id);settings[key]=type==='boolean'?input.checked:type==='number'?Number(input.value===''?fallback:input.value):(input.value||fallback);}
 generationZones.update(settings.tasks,settings.randomize_layout);settings.object_zones=generationZones.settings();settings.extra_objects=generationZones.extras();if(settings.randomize_layout)Object.assign(settings,generationZones.layoutSettings());
 if(generation.parent)settings.parent_job_id=generation.parent.id;
 return settings;
}
function restoreGenerationSettings(options){
 // Older jobs used conservative timing before the recorded profile was introduced.
 const saved={...options,timing_profile:options.timing_profile||'conservative'};
 for(const [key,[id,fallback,type]] of Object.entries(generationFields)){const input=$('#generation-'+id),value=saved[key]??fallback;if(type==='boolean')input.checked=value;else input.value=String(value);}
 for(const input of $$('[name="generation-task"]'))input.checked=(saved.tasks||[]).includes(input.value);
 generationZones.restore(saved.object_zones||{},saved.extra_objects??LEGACY_EXTRA_OBJECTS);generationZones.restoreLayout(options);
}
function beginGenerationExtension(){
 const job=generation.job;if(job?.status!=='completed'||job.export?.valid!==true||generation.submitting)return;
 invalidateGenerationStills();
 generation.parent={id:job.id,name:job.name,count:job.export.total_episodes??job.progress?.accepted??(job.episodes||[]).filter(episode=>episode.status==='accepted').length};
 const background=job.options?.background_id||'calibrated-real';
 if(!generation.backgrounds.some(asset=>asset.id===background))generation.backgrounds.push({id:background,name:background,mode:'uploaded'});
 renderGenerationBackgrounds(background);
 restoreGenerationSettings(job.options||{});
 $('#generation-name').value=(job.name||'so101-synthetic').slice(0,55)+'-extended';
 $('#generation-seed').value=String(job.next_seed??((job.options?.seed??42)+(job.progress?.attempts??0)+1));
 updateGenerationBackground();updateGenerationForm();setGenerationTab('build');
}
// Real replay datasets can only grow with more recordings, so extension reopens the replay builder on their source.
async function beginReplayExtension(job){
 const source=job.options?.source_id;if(job?.status!=='completed'||job.export?.valid!==true||!source)return;
 await chooseDataset(source);if(state.selectedDataset!==source)return;
 video.pause();await replayBuilder.open(source,0,job);$('#replay-builder').scrollIntoView({behavior:'smooth',block:'start'});
}
function endGenerationExtension(){generation.parent=null;invalidateGenerationStills();updateGenerationForm();}
function renderGenerationBackgrounds(selected=$('#generation-background').value||'calibrated-real'){
 const select=$('#generation-background');select.replaceChildren();
 for(const asset of generation.backgrounds)select.append(new Option(asset.name,asset.id));
 select.value=selected;updateGenerationBackground();
}
function updateGenerationBackground(){
 const id=$('#generation-background').value||'calibrated-real',asset=generation.backgrounds.find(item=>item.id===id),image=$('#generation-background-image');
 image.classList.toggle('hidden',!asset?.url);if(asset?.url)image.src=asset.url;else image.removeAttribute('src');
 $('#generation-background-note').textContent=(id==='calibrated-real'?'Calibrated real workcell imagery with moving-camera projection.':'Uploaded image: uncalibrated, mapped onto the workcell background plane. Robot and objects remain rendered geometry.')+($('#generation-real-background').checked?'':' Enable “Use selected background image” to apply it.');
}
async function loadGenerationBackgrounds(){
 if(generation.assetsLoaded)return;
 try{const result=await api('/api/generation-assets');generation.backgrounds=result.backgrounds||generation.backgrounds;generation.assetsLoaded=true;renderGenerationBackgrounds();$('#generation-background-status').textContent='';}
 catch(error){$('#generation-background-status').textContent='Unable to load the background library: '+error.message;}
}
async function uploadGenerationBackground(){
 const input=$('#generation-background-upload'),file=input.files?.[0];if(!file||generation.uploading)return;
 if(!['image/png','image/jpeg','image/webp'].includes(file.type)){$('#generation-background-status').textContent='Choose a PNG, JPEG or WebP image.';input.value='';return;}
 generation.uploading=true;invalidateGenerationStills();updateGenerationForm();input.disabled=true;$('#generation-background-status').textContent='Uploading background…';
 try{
  const result=await api('/api/generation-assets/upload',file,{headers:{'X-Filename':encodeURIComponent(file.name)}}),asset=result.background;
  if(!asset?.id)throw Error('The background could not be saved.');
  generation.backgrounds=generation.backgrounds.filter(item=>item.id!==asset.id);generation.backgrounds.push(asset);
  $('#generation-real-background').checked=true;renderGenerationBackgrounds(asset.id);invalidateGenerationStills();$('#generation-background-status').textContent='Background saved to your local library.';
 }catch(error){$('#generation-background-status').textContent='Upload failed: '+error.message;}
 finally{generation.uploading=false;input.disabled=false;input.value='';updateGenerationForm();}
}
function invalidateGenerationStills(){
 const preview=generation.preview,hadPreview=preview.signature!==null;
 preview.request++;clearTimeout(preview.timer);preview.timer=null;preview.busy=false;preview.signature=null;
 $('#generation-still-results').replaceChildren();
 if(hadPreview)$('#generation-still-status').textContent='Settings changed. Preview again to see the current appearance and layout.';
}
function renderGenerationStills(result,settings){
 const list=$('#generation-still-results');list.replaceChildren();
 for(const item of result.images||[]){
  const row=document.createElement('article'),heading=document.createElement('h4'),pair=document.createElement('div');
  row.className='generation-still-row';pair.className='generation-still-pair';
  heading.textContent=`${Number(item.index)+1} · ${generationTaskName(item.task_id,settings)} · Seed ${item.seed}`;
  for(const [key,label] of [['top_url','Top camera'],['wrist_url','Wrist camera']]){const figure=document.createElement('figure'),caption=document.createElement('figcaption'),image=document.createElement('img');caption.textContent=label;image.src=item[key];image.alt=`${label}: ${generationTaskName(item.task_id,settings)}`;image.loading='lazy';figure.append(caption,image);pair.append(figure);}
  row.append(heading,pair);list.append(row);
 }
 list.dataset.settingsHash=result.settings_hash||'';
 $('#generation-still-status').textContent=`${(result.images||[]).length} approximate scenes · current settings. Motion and physics are validated during generation.`;
}
async function startGenerationStills(){
 if(generation.preview.busy||generation.submitting||generation.uploading||generation.busy||!generation.loaded||!$('#generation-form').reportValidity())return;
 const settings=collectGenerationSettings();if(!settings.tasks.length||generationZones.error){updateGenerationForm();return;}
 invalidateGenerationStills();const preview=generation.preview,request=preview.request,signature=JSON.stringify(settings);
 preview.signature=signature;preview.busy=true;updateGenerationForm();$('#generation-still-status').textContent='Preparing five appearance and layout previews…';
 const current=()=>request===preview.request&&signature===JSON.stringify(collectGenerationSettings());
 const finish=()=>{preview.busy=false;preview.timer=null;updateGenerationForm();};
 const receive=async(result,attempt=0)=>{
  if(!current())return;
  if(result.status==='completed'){renderGenerationStills(result,settings);finish();return;}
  if(result.status==='failed'||result.status==='cancelled')throw Error(result.error||'Preview stopped. Preview the current settings again.');
  if(!['queued','running'].includes(result.status))throw Error('The preview returned an unexpected status.');
  if(attempt>=120)throw Error('Preview is taking too long. Please try again shortly.');
  $('#generation-still-status').textContent=result.status==='queued'?'Preview queued…':'Rendering five appearance and layout previews…';
  preview.timer=setTimeout(async()=>{try{if(!current())return;await receive(await api('/api/generation-preview/'+encodeURIComponent(result.id)),attempt+1);}catch(error){if(current()){$('#generation-still-status').textContent='Preview unavailable: '+error.message;finish();}}},1500);
 };
 try{await receive(await api('/api/generation-preview',settings));}
 catch(error){if(current()){$('#generation-still-status').textContent='Preview unavailable: '+error.message;finish();}}
}
function updateGenerationForm(){
 $('#generation-layout-strategy').disabled=!$('#generation-layout').checked;
 if(!$('#generation-layout').checked)$('#generation-layout-strategy').value='uniform';
 const tasks=[...$$('[name="generation-task"]:checked')].map(input=>input.value),count=Number($('#generation-count').value);
 generationZones.update(tasks,$('#generation-layout').checked);
 for(const input of $$('[name="generation-task"]'))input.nextElementSibling.textContent=generationTaskName(input.value,{unstack_top:$('#generation-unstack').value});
 $('#generation-unstack-field').classList.toggle('hidden',!tasks.includes('unstack'));
 $('#generation-speed-value').textContent=Math.round(Number($('#generation-speed').value)*100)+'%';
 const total=tasks.length*count;
 $('#generation-estimate').textContent=`${tasks.length} task${tasks.length===1?'':'s'} · ${Number.isFinite(total)?total:0} requested episode${total===1?'':'s'}`;
 $('#generation-submit').disabled=!generation.loaded||generation.submitting||generation.busy||generation.uploading||generation.preview.busy||!tasks.length||state.restarting||!!generationZones.error;
 $('#generation-preview-button').disabled=$('#generation-submit').disabled;
 $('#generation-preview-button').textContent=generation.preview.busy?'Rendering preview…':'Preview 5 scenes';
 $('#generation-extension').classList.toggle('hidden',!generation.parent);
 $('#generation-build-title').textContent=generation.parent?'Add episodes':'Build a dataset';
 if(generation.parent){$('#generation-extension-title').textContent='Extend '+generation.parent.name;$('#generation-extension-summary').textContent=`${generation.parent.count} existing + ${Number.isFinite(total)?total:0} new requested = ${generation.parent.count+(Number.isFinite(total)?total:0)} episodes if all pass validation.`;}
 $('#generation-submit').textContent=generation.submitting?'Starting generation…':generation.parent?'Generate extended dataset':'Generate dataset';
 $('#generation-form-status').textContent=generation.submitting?'Submitting your generation settings…':generation.uploading?'Saving your background before generation…':generation.preview.busy?'Rendering the preview before generation…':generation.busy?'A generation job is active. Review its progress or cancel it in Visualize data.':!tasks.length?'Select at least one task.':generationZones.error||(!generation.loaded?'Connecting to the generation service…':'Generation runs locally in simulation.');
}
function generationURL(){const params=new URLSearchParams({page:'generate',tab:generation.tab});if(generation.job?.id)params.set('job',generation.job.id);if(generation.tab==='visualize')params.set('episode',String(generation.episodeIndex));history.replaceState(null,'','/?'+params);}
function setGenerationTab(tab){
 generation.tab=tab==='visualize'?'visualize':'build';
 for(const name of ['build','visualize']){const selected=name===generation.tab;$('#generation-'+name).classList.toggle('hidden',!selected);$('#generation-tab-'+name).setAttribute('aria-selected',String(selected));$('#generation-tab-'+name).tabIndex=selected?0:-1;}
 if(generation.tab==='build')generationViewer.suspend();else renderGenerationPreview();
 if(state.page==='generate')generationURL();
}
function renderGenerationPreview(){
 if(state.page==='generate'&&generation.tab==='visualize'){generationViewer.load(generation.job,generation.episodeIndex);generationCoverage.load(generation.job,generation.episodeIndex);}
 for(const button of $$('#generation-episode-list button'))button.setAttribute('aria-pressed',String(Number(button.dataset.index)===generation.episodeIndex));
}
function selectGenerationEpisode(index){generation.episodeIndex=Number(index);generationURL();renderGenerationPreview();}
function renderGenerationEpisodes(){
 const episodes=generation.job?.episodes||[],signature=JSON.stringify([generation.job?.id,episodes]);
 if(signature===generation.episodesSignature)return;
 generation.episodesSignature=signature;
 const accepted=episodes.map((episode,index)=>({...episode,index})).filter(episode=>episode.status==='accepted');
 if(!accepted.some(episode=>episode.index===generation.episodeIndex))generation.episodeIndex=accepted[0]?.index??0;
 $('#generation-episode-count').textContent=`${accepted.length} accepted`;
 const buttons=$('#generation-episode-list');buttons.replaceChildren();
 for(const episode of accepted){const button=document.createElement('button');button.type='button';button.className='generation-episode-item';button.dataset.index=episode.index;button.textContent=`${String(episode.index+1).padStart(3,'0')} · ${episode.instruction||generationTaskName(episode.task_id)}`;button.onclick=()=>selectGenerationEpisode(episode.index);buttons.append(button);}
 if(!accepted.length){const empty=document.createElement('p');empty.className='generation-help';empty.textContent='Accepted episodes will appear here.';buttons.append(empty);}
 renderGenerationPreview();
}
function renderGenerationBatchSettings(){
 const job=generation.job,options=job?.options,details=$('#generation-batch-settings'),values=$('#generation-batch-values');
 details.classList.toggle('hidden',!job);
 $('#generation-batch-note').textContent=job?.parent_job_id?'Saved settings for the new episodes in this revision. Inherited episodes retain their original settings.':'Saved batch settings · read only. The form creates a new batch with its current values.';
 if(details.dataset.jobId!==(job?.id||'')){details.open=false;details.dataset.jobId=job?.id||'';}
 values.replaceChildren();if(!job)return;
 const add=(label,value)=>{const term=document.createElement('dt'),description=document.createElement('dd');term.textContent=label;description.textContent=value;values.append(term,description);};
 if(!options){add('Saved settings','Not available for this batch.');return;}
 const tasks=Array.isArray(options.tasks)?options.tasks:[];
 add('Tasks',tasks.length?tasks.map(id=>generationTaskName(id,options)).join(' · '):'Not recorded');
 add('Episodes per task',options.episodes_per_task??'Not recorded');
 if(options.target_counts)add('Target totals',Object.entries(options.target_counts).map(([task,total])=>`${generationTaskName(task,options)} ${total}`).join(' · '));
 add('Layout distribution',options.layout_strategy==='balanced'?'Balanced workspace zones':'Uniform proposals');
 if(options.randomize_layout!==false)for(const key of activeZoneObjects(tasks,options.extra_objects??LEGACY_EXTRA_OBJECTS))add(ZONE_OBJECTS[key]+' initial zones',zoneSummary(options.object_zones?.[key]));
 if(options.randomize_layout!==false){const spread=options.zone_spread_mm,yaw=options.object_yaw_range_deg;
  for(const key of activeZoneObjects(tasks,options.extra_objects??LEGACY_EXTRA_OBJECTS)){const v=spread!==null&&typeof spread==='object'?spread[key]:spread;add(ZONE_OBJECTS[key]+' spread',v==null?'Whole zone':`${v} mm radius`);}
  for(const key of ['red_cube','green_cube','bowl'].filter(k=>activeZoneObjects(tasks,options.extra_objects??LEGACY_EXTRA_OBJECTS).includes(k))){const v=yaw!==null&&typeof yaw==='object'?yaw[key]:yaw;add(ZONE_OBJECTS[key]+' rotation',`${v??(key==='bowl'?0:11.46)}°`);}}
 {const extras=options.extra_objects??LEGACY_EXTRA_OBJECTS;add('Extra objects in scene',options.extra_objects===undefined?'Both cubes in every task (earlier default)':extras.length?extras.map(key=>ZONE_OBJECTS[key]).join(' · '):'Task objects only');}
 for(const [task,coverage] of Object.entries(job.coverage||{}))add(generationTaskName(task,options),`Pick zones: ${coverage.source_counts.join(' / ')} · Destination zones: ${coverage.target_counts.join(' / ')}`);
 add('Random seed',options.seed??'Not recorded');
 const motion=motionLabels[options.motion_technique]||pretty(options.motion_technique)||'Not recorded';
 add('Motion timing',generationTimingName(options));
 add('Motion / travel speed',motion+(typeof options.speed_scale==='number'?` · ${Math.round(options.speed_scale*100)}%`:''));
 const variations=[['randomize_layout','Object layout'],['real_background','Real-image background'],['real_image_processing','Match real camera appearance'],['visual_randomization','Conservative appearance'],['camera_jitter','Camera jitter'],['sensor_augmentation','Sensor augmentation'],['physics_randomization','Physics randomization'],['lighting_randomization','Lighting randomization']].filter(([key])=>options[key]===true).map(([,label])=>label);
 add('Enabled variations',variations.length?variations.join(' · '):'None');
 add('Background',options.background_id||'calibrated-real');
 add('Lighting',`Intensity ${options.light_intensity??1} · direction ${options.light_direction??0}° · elevation ${options.light_elevation??55}° · spread ${options.light_spread??45}° · shadow ${options.shadow_strength??.5}`);
 add('Unstack top cube',tasks.includes('unstack')?(options.unstack_top==='red'?'Red':options.unstack_top==='green'?'Green':'Not recorded'):'Not applicable');
 add('Requested validation fraction',typeof options.validation_fraction==='number'?`${Math.round(options.validation_fraction*100)}%`:'Not recorded');
}
function renderGenerationInfo(){
 const job=generation.job,info=$('#generation-dataset-info');info.replaceChildren();if(!job)return;
 const add=(label,value)=>{const item=document.createElement('div'),term=document.createElement('dt'),description=document.createElement('dd');term.textContent=label;description.textContent=value;item.append(term,description);info.append(item);};
 const summary=job.summary||{},exported=job.export||{},frames=summary.total_frames??exported.total_frames;
 const duration=summary.total_duration_s??(frames&&summary.fps?frames/summary.fps:null);
 const created=typeof job.created_at==='number'?new Date(job.created_at*1000):new Date(job.created_at);
 add('Created',Number.isNaN(created.getTime())?'Not recorded':created.toLocaleString());
 add('Robot / format','SO-101 · '+(summary.format||'LeRobot v3'));
 add('Frames / duration',frames!=null?`${Number(frames).toLocaleString()} frames${duration!=null?' · '+Number(duration).toFixed(1)+' s':''}`:'Available after generation');
 add('Cameras',Array.isArray(summary.cameras)?summary.cameras.map(pretty).join(' + '):'Top + Wrist');
 if(summary.fps)add('Frame rate',`${summary.fps} Hz`);
 if(Array.isArray(summary.resolution))add('Image size',summary.resolution.join(' × ')+' px');
 add('Motion timing',job.source_type==='real_replay'?'Original recorded actions and timing':generationTimingName({timing_profile:summary.timing_profile??job.options?.timing_profile}));
 if(job.source_type==='real_replay')add('Source',job.options?.source_id||'Recorded real dataset');
 if(job.parent_job_id)add('Extended from',job.parent_job_id+` · ${job.lineage?.inherited_count??job.progress?.inherited_count??0} inherited episodes`);
 if(job.next_seed!=null)add('Next available seed',job.next_seed);
 for(const row of generationTaskCounts(job))add(generationTaskLabel(job,row.task_id,generationTaskName),`${row.accepted}${row.target!=null?' / '+row.target:''} accepted episodes`);
 const instructions=(job.episodes||[]).filter(episode=>episode.status==='accepted').map(episode=>episode.instruction||generationTaskName(episode.task_id,job.options));
 // Pending tasks remain visible before completion; completed datasets describe accepted episodes.
 if(job.status!=='completed'||!Array.isArray(job.episodes))instructions.push(...(job.options?.tasks||[]).map(id=>generationTaskName(id,job.options)));
 if(instructions.length)add('Tasks',[...new Set(instructions)].join(' '));
}
function dismissGenerationDelete(){generation.deleteId=null;$('#generation-delete-prompt').classList.add('hidden');}
function renderGenerationTrash(items){
 const root=$('#generation-trash-list'),signature=JSON.stringify(items);if(root.dataset.signature===signature)return;root.dataset.signature=signature;root.replaceChildren();
 $('#generation-trash-summary').textContent=`Trash · ${items.length} deleted datasets`;
 for(const item of items){const row=document.createElement('div'),label=document.createElement('span'),restore=document.createElement('button');row.className='generation-trash-item';label.textContent=`${item.name} · ${item.episodes} episodes`;restore.className='button small';restore.type='button';restore.textContent='Restore';restore.onclick=async()=>{restore.disabled=true;try{await api(`/api/generation-trash/${encodeURIComponent(item.id)}/restore`,{});await refreshGeneration();toast('Dataset restored.');}catch(error){toast(error.message,true);restore.disabled=false;}};row.append(label,restore);root.append(row);}
 if(!items.length){const empty=document.createElement('p');empty.textContent='No deleted datasets.';root.append(empty);}
}
function renderGenerationJob(){
 const job=generation.job,progress=job?.progress||{},active=generationActive(job);
 $('#generation-job-title').textContent=job?.name||'Dataset information';
 $('#generation-job-status').textContent=pretty(job?.status||'ready');
 $('#generation-job-status').classList.toggle('failed',['failed','cancelled'].includes(job?.status));
 const phase=pretty(job?.phase||job?.status),taskInstruction=progress.task_id?generationTaskName(progress.task_id):'';
 $('#generation-phase').textContent=job?phase+(active&&taskInstruction&&!phase.includes(taskInstruction)?' · '+taskInstruction:''):'Choose a generated dataset or build a new one.';
 for(const name of ['accepted','rejected','requested','attempts'])$('#generation-'+name).textContent=progress[name]??0;
 $('#generation-progress').max=Math.max(1,Number(progress.requested)||1);$('#generation-progress').value=Number(progress.accepted)||0;
 $('#generation-cancel').classList.toggle('hidden',!active);$('#generation-cancel').disabled=generation.cancelling;$('#generation-cancel').textContent=generation.cancelling?'Cancelling…':'Cancel generation';
 $('#generation-delete').classList.toggle('hidden',!job||active);$('#generation-delete').disabled=generation.deleting||generation.busy;
 if(generation.deleteId!==job?.id)dismissGenerationDelete();
 const ready=job?.status==='completed'&&job.export?.valid===true&&!!job.export?.url,download=$('#generation-download');
 download.classList.toggle('hidden',!ready);
 $('#generation-add-episodes').classList.toggle('hidden',job?.kind==='preview'||job?.status!=='completed'||job.export?.valid!==true);
 $('#generation-add-episodes').disabled=generation.submitting||generation.uploading;
 if(ready)download.href=`/api/generation/${encodeURIComponent(job.id)}/download`;else download.removeAttribute('href');
 const split=job?.export?.splits,splitNote=Array.isArray(split?.train)&&Array.isArray(split?.val)?` · ${split.train.length} training / ${split.val.length} validation${split.val.length?'':' (no validation split)'}`:'';
 const exportedCount=job?.export?.total_episodes??progress.accepted??0;
 $('#generation-export-note').textContent=ready?`${exportedCount} episode${exportedCount===1?'':'s'} · ${job.export.total_frames??0} frames${splitNote} · export validation passed`:job?.status==='failed'?'Generation failed. Review the errors below.':job?.status==='cancelled'?'Generation cancelled. No completed export is available.':job?.status==='completed'?'No validated export is available for this job.':'A download appears after the export passes validation.';
 const errors=$('#generation-errors');errors.replaceChildren();for(const error of job?.errors||[]){const item=document.createElement('li');item.textContent=typeof error==='string'?error:error.message||JSON.stringify(error);errors.append(item);}errors.classList.toggle('hidden',!errors.children.length);
 renderGenerationInfo();renderGenerationBatchSettings();renderGenerationEpisodes();updateGenerationForm();
}
async function refreshGeneration(){
 const data=await api('/api/generation');generation.jobs=data.jobs||[];generation.busy=!!data.busy;generation.loaded=true;renderGenerationTrash(data.trash||[]);
 for(const task of data.tasks||[]){const input=[...$$('[name="generation-task"]')].find(item=>item.value===task.id);if(input){if(task.instruction&&task.id!=='unstack')generationTaskInstructions[task.id]=task.instruction;}}
 const select=$('#generation-dataset'),signature=JSON.stringify(generation.jobs.map(job=>[job.id,job.name,job.status,job.progress?.accepted]));
 if(select.dataset.signature!==signature){select.dataset.signature=signature;select.replaceChildren();if(!generation.jobs.length)select.append(new Option('No generated datasets yet',''));for(const job of generation.jobs)select.append(new Option(`${job.name} · ${job.progress?.accepted??0} episodes · ${job.status}`,job.id));}
 select.value=generation.job?.id||'';updateGenerationForm();return data;
}
async function loadGenerationJob(id,episodeIndex=0){
 generationViewer.suspend();dismissGenerationDelete();
 const request=++generation.request;
 $('#generation-phase').textContent='Loading generation job…';
 const job=await api('/api/generation/'+encodeURIComponent(id));
 if(request!==generation.request||state.page!=='generate')return;
 generation.job=job;generation.episodeIndex=Number.isInteger(episodeIndex)&&episodeIndex>=0?episodeIndex:0;generation.episodesSignature='';
 $('#generation-dataset').value=job.id;renderGenerationJob();generationURL();
}
async function pollGeneration(){
 clearTimeout(generation.timer);if(state.page!=='generate'||state.restarting)return;
 try{
  const request=generation.request,id=generation.job?.id;
  await refreshGeneration();
  if(state.page!=='generate'||request!==generation.request)return;
  if(id){const job=await api('/api/generation/'+encodeURIComponent(id));if(state.page!=='generate'||request!==generation.request)return;generation.job=job;renderGenerationJob();}
 }catch(error){if(state.page==='generate')$('#generation-'+(generation.tab==='visualize'?'phase':'form-status')).textContent='Unable to refresh generation: '+error.message+'. Retrying…';}
 finally{if(state.page==='generate'&&!state.restarting)generation.timer=setTimeout(pollGeneration,2500);}
}
async function showGeneration(id,tab=generation.tab,episodeIndex=0){
 if(state.busy||state.selecting||state.robot.running)return;
 resetViewer();state.page='generate';state.historyMode=false;generation.request++;clearTimeout(generation.timer);
 history.replaceState(null,'',id?'/?page=generate&job='+encodeURIComponent(id):'/?page=generate');applyLayout();updateControls();updateGenerationForm();setGenerationTab(tab);
 try{await refreshGeneration();await loadGenerationBackgrounds();if(state.page!=='generate')return;const selected=id||generation.job?.id||generation.jobs.find(generationActive)?.id||generation.jobs[0]?.id;if(selected)await loadGenerationJob(selected,episodeIndex);else renderGenerationJob();}
 catch(error){$('#generation-form-status').textContent='Unable to load generation: '+error.message;toast(error.message,true);}
 finally{if(state.page==='generate')generation.timer=setTimeout(pollGeneration,2500);}
}
$('#nav-generate').onclick=()=>showGeneration();
$('#generation-form').oninput=event=>{if(event?.target?.id==='generation-background-upload')return;invalidateGenerationStills();updateGenerationBackground();updateGenerationForm();};
 $('#generation-add-episodes').onclick=()=>generation.job?.source_type==='real_replay'?beginReplayExtension(generation.job):beginGenerationExtension();
 $('#generation-new-dataset').onclick=endGenerationExtension;
 $('#generation-background-upload').onchange=uploadGenerationBackground;
 $('#generation-preview-button').onclick=startGenerationStills;
$('#generation-form').onsubmit=async event=>{
 event.preventDefault();if(generation.submitting||generation.busy||generation.uploading||generation.preview.busy||!$('#generation-form').reportValidity())return;
 const tasks=[...$$('[name="generation-task"]:checked')].map(input=>input.value);if(!tasks.length){updateGenerationForm();return;}
 const name=$('#generation-name').value.trim();if(!name){$('#generation-name').focus();toast('Enter a dataset name.',true);return;}
 const payload=collectGenerationSettings();
 if(generationZones.error){updateGenerationForm();return;}
 try{generation.submitting=true;updateGenerationForm();const job=await api('/api/generation',payload);generation.parent=null;invalidateGenerationStills();generation.busy=generationActive(job);if(state.page==='generate'){generation.request++;generation.job=job;generation.episodeIndex=0;generation.episodesSignature='';renderGenerationJob();setGenerationTab('visualize');}await refreshGeneration();}
 catch(error){toast(error.message,true);}
 finally{generation.submitting=false;updateGenerationForm();}
};
$('#generation-cancel').onclick=async()=>{
 const id=generation.job?.id;if(!id||generation.cancelling||!generationActive(generation.job))return;
 try{generation.cancelling=true;renderGenerationJob();const job=await api(`/api/generation/${encodeURIComponent(id)}/cancel`,{});if(generation.job?.id===id){generation.job=job;renderGenerationJob();}await refreshGeneration();}
 catch(error){toast(error.message,true);}
 finally{generation.cancelling=false;renderGenerationJob();}
};
$('#generation-delete').onclick=()=>{const job=generation.job;if(!job||generationActive(job)||generation.deleting)return;generation.deleteId=job.id;$('#generation-delete-name').textContent=`Delete ${job.name}?`;$('#generation-delete-prompt').classList.remove('hidden');};
$('#generation-delete-dismiss').onclick=dismissGenerationDelete;
$('#generation-delete-confirm').onclick=async()=>{
 const id=generation.deleteId;if(!id||id!==generation.job?.id||generation.deleting)return;
 const request=++generation.request;generation.deleting=true;generationViewer.suspend();$('#generation-delete-confirm').disabled=true;
 try{await api(`/api/generation/${encodeURIComponent(id)}/delete`,{});if(generation.job?.id===id){generation.job=null;generation.episodesSignature='';if(generation.parent?.id===id)generation.parent=null;renderGenerationJob();}dismissGenerationDelete();await refreshGeneration();if(state.page==='generate'&&request===generation.request){if(generation.jobs.length)await loadGenerationJob(generation.jobs[0].id);else{renderGenerationJob();generationURL();}}toast('Dataset moved to Trash.');}
 catch(error){toast(error.message,true);}
 finally{generation.deleting=false;$('#generation-delete-confirm').disabled=false;renderGenerationJob();}
};
$('#generation-refresh').onclick=()=>pollGeneration();
$('#generation-dataset').onchange=()=>loadGenerationJob($('#generation-dataset').value).catch(error=>toast(error.message,true));
for(const name of ['build','visualize']){const button=$('#generation-tab-'+name);button.onclick=()=>setGenerationTab(name);button.onkeydown=event=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();const other=event.key==='Home'?'build':event.key==='End'?'visualize':name==='build'?'visualize':'build';setGenerationTab(other);$('#generation-tab-'+other).focus();}};}
(async()=>{try{const data=await api('/api/bootstrap');state.token=data.token;state.instance=data.instance;$('#restart-server').disabled=!data.restart_available;const catalog=await api('/api/datasets');state.datasets=catalog.datasets;$('#dataset-count').textContent=state.datasets.length;for(const [i,j] of data.joints.entries())$('#joint').append(new Option(pretty(j),i));renderDatasets();await refreshRuns();const params=new URL(location.href).searchParams,requested=params.get('run');if(params.get('page')==='generate')await showGeneration(params.get('job'),params.get('tab'),Number(params.get('episode')||0));else if(requested&&state.runs.some(r=>r.id===requested))await loadRun(requested);else if(params.get('page')==='objects')await showObjects();else if(params.get('page')==='new')await newRun();else if(params.get('page')==='history'||requested)await showHistory();else showDatasets();poll();livePreview();}catch(e){toast(e.message,true);}})();
