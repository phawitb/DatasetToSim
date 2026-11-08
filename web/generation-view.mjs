import {EpisodePlayback} from './generation-playback.mjs';
const $=selector=>document.querySelector(selector);
export class GenerationVisualizer{
 constructor(api){
  this.api=api;this.active=false;this.version=0;this.key=null;this.data=null;this.frame=0;this.charts=[];
  this.video=$('#generation-video');
  this.player=new EpisodePlayback(this.video,{onFrame:(frame)=>this.paint(frame),onState:state=>this.controls(state),onError:message=>{$('#generation-preview-status').textContent=message;this.key=null;}});
  $('#generation-play').onclick=()=>this.player.toggle();
  $('#generation-timeline').oninput=event=>this.player.seekFrame(Number(event.target.value));
  $('#generation-timeline').onpointerdown=()=>this.player.pause();
  $('#generation-timeline').onkeydown=event=>{if(['ArrowLeft','ArrowRight','Home','End','PageUp','PageDown'].includes(event.key))this.player.pause();};
  $('#generation-previous').onclick=()=>this.player.seekFrame(this.frame-1);
  $('#generation-next').onclick=()=>this.player.seekFrame(this.frame+1);
  $('#generation-play-speed').onchange=event=>this.player.setRate(event.target.value);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)this.player.pause();});
  this.clear();
 }
 suspend(){this.active=false;this.player.pause();this.version++;if(!this.data)this.key=null;}
 clear(){
  this.data=null;this.loaded=false;this.frame=0;this.charts=[];this.player.setEpisode(null);
  $('#generation-joint-charts').replaceChildren();$('#generation-frame-label').textContent='FRAME —';$('#generation-time').textContent='0.00 / 0.00 s';$('#generation-timeline').value=0;
  for(const camera of ['top','wrist']){const canvas=$('#generation-canvas-'+camera),ctx=canvas.getContext('2d');canvas.removeAttribute('data-frame');ctx.fillStyle='#edf2e8';ctx.fillRect(0,0,640,480);ctx.fillStyle='#819079';ctx.font='18px sans-serif';ctx.textAlign='center';ctx.fillText('Select an accepted episode',320,240);}
  this.controls({ready:false,paused:true});
 }
 controls({ready,paused}){
  if(this.data&&this.video.readyState>=2)this.loaded=true;
  const loaded=!!this.data&&this.loaded;
  $('#generation-play').disabled=!ready||!this.data;$('#generation-play').textContent=paused?'▶ Play':'Ⅱ Pause';$('#generation-play').setAttribute('aria-label',paused?'Play generated episode':'Pause generated episode');
  for(const id of ['previous','next','timeline'])$('#generation-'+id).disabled=!loaded;
 }
 async load(job,index){
  this.active=true;
  const episode=job?.episodes?.[index],key=episode?.status==='accepted'?`${job.id}:${index}`:null;
  if(key&&key===this.key)return;
  const version=++this.version;this.key=key;this.clear();
  if(!key){$('#generation-visual-title').textContent='No accepted episodes yet';$('#generation-preview-status').textContent='Select a dataset with accepted episodes, or build a new dataset.';return;}
  $('#generation-visual-title').textContent=`Episode ${String(index+1).padStart(3,'0')}`;
  $('#generation-preview-status').textContent='Preparing synchronized cameras and robot data…';
  try{
   let data;
   for(let attempt=0;attempt<150;attempt++){
    data=await this.api(`/api/generation/${encodeURIComponent(job.id)}/episodes/${index}`);
    if(version!==this.version||!this.active)return;
    if(data.status==='ready')break;
    await new Promise(resolve=>setTimeout(resolve,1000));
   }
   if(data.status!=='ready')throw new Error('Preview preparation timed out. Select the episode again to retry.');
   const url=new URL(data.video_url,location.origin);
   if(url.origin!==location.origin||!url.pathname.startsWith('/media/generated/'))throw new Error('Invalid preview source');
   this.data=data;this.buildCharts();$('#generation-timeline').max=data.frames-1;
   $('#generation-preview-status').textContent=`${data.instruction} · ${data.frames} frames · ${data.fps} Hz · Synchronized`;
   this.player.setEpisode({url:url.href,frames:data.frames,fps:data.fps});this.player.setRate($('#generation-play-speed').value);
  }catch(error){if(version!==this.version||!this.active)return;this.key=null;this.clear();$('#generation-preview-status').textContent=error.message;}
 }
 buildCharts(){
  const root=$('#generation-joint-charts');root.replaceChildren();this.charts=[];
  const data=this.data,n=data.frames;
  data.joints.forEach((name,j)=>{
   const card=document.createElement('article');card.className='generation-joint';
   const heading=document.createElement('h4');heading.textContent=name.replace('.pos','').replaceAll('_',' ');
   const values=document.createElement('div');values.className='generation-joint-values';
   const state=document.createElement('span'),action=document.createElement('span');state.className='generation-state-key';action.className='generation-action-key';values.append(state,action);
   const all=[...data.state.map(row=>row[j]),...data.action.map(row=>row[j])];let low=Math.min(...all),high=Math.max(...all);const pad=Math.max(1,(high-low)*.12);low-=pad;high+=pad;
   const x=i=>30+(i/Math.max(1,n-1))*240,y=value=>83-(value-low)/(high-low)*62;
   const path=rows=>rows.map((row,i)=>`${i?'L':'M'}${x(i).toFixed(2)},${y(row[j]).toFixed(2)}`).join(' ');
   const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 280 108');svg.setAttribute('role','img');svg.setAttribute('aria-label',heading.textContent+' state and action trajectory');
   svg.innerHTML=`<path d="M30 21H270M30 52H270M30 83H270" stroke="#e5ebdf" fill="none"/><text x="25" y="24" text-anchor="end">${high.toFixed(0)}</text><text x="25" y="85" text-anchor="end">${low.toFixed(0)}</text><text x="30" y="102">0 s</text><text x="270" y="102" text-anchor="end">${((n-1)/data.fps).toFixed(1)} s</text><path d="${path(data.state)}" stroke="#3986ae" stroke-width="1.6" fill="none"/><path d="${path(data.action)}" stroke="#9473ba" stroke-width="1.6" stroke-dasharray="4 3" fill="none"/><line class="joint-cursor" x1="30" x2="30" y1="17" y2="87" stroke="#284e39" stroke-width="1.4"/><circle class="joint-state-point" r="3" fill="#3986ae"/><circle class="joint-action-point" r="3" fill="#9473ba"/>`;
   card.append(heading,values,svg);root.append(card);
   this.charts.push({state,action,x,y,cursor:svg.querySelector('.joint-cursor'),statePoint:svg.querySelector('.joint-state-point'),actionPoint:svg.querySelector('.joint-action-point')});
  });
 }
 paint(frame){
  if(!this.active||!this.data||this.video.readyState<2)return;
  this.frame=frame;const data=this.data;
  for(const [index,camera]of ['top','wrist'].entries()){
   const canvas=$('#generation-canvas-'+camera);canvas.getContext('2d').drawImage(this.video,index*640,0,640,480,0,0,640,480);canvas.dataset.frame=frame;
  }
  $('#generation-frame-label').textContent=`FRAME ${String(frame+1).padStart(3,'0')} / ${data.frames}`;
  $('#generation-time').textContent=`${(frame/data.fps).toFixed(2)} / ${data.duration.toFixed(2)} s`;
  $('#generation-timeline').value=frame;$('#generation-joint-charts').dataset.frame=frame;
  this.charts.forEach((chart,j)=>{
   const state=data.state[frame][j],action=data.action[frame][j],unit=j===5?'':'°',x=chart.x(frame);
   chart.state.textContent=`State ${state.toFixed(1)}${unit}`;chart.action.textContent=`Action ${action.toFixed(1)}${unit}`;
   chart.cursor.setAttribute('x1',x);chart.cursor.setAttribute('x2',x);
   for(const [point,value]of [[chart.statePoint,state],[chart.actionPoint,action]]){point.setAttribute('cx',x);point.setAttribute('cy',chart.y(value));}
  });
 }
}
