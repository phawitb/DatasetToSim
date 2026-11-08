/** Saved initial scene positions; this view never derives positions from videos. */
export function generationTaskCounts(job){
 const counts=new Map();
 for(const task of job?.options?.tasks||[])counts.set(task,0);
 for(const episode of job?.episodes||[])if(episode.status==='accepted')counts.set(episode.task_id,(counts.get(episode.task_id)||0)+1);
 return [...counts].map(([task_id,accepted])=>({task_id,accepted,target:job?.options?.target_counts?.[task_id]??null}));
}
export function generationTaskLabel(job,task,fallback=id=>id.replaceAll('_',' ')){
 const instructions=[...new Set((job?.episodes||[]).filter(e=>e.status==='accepted'&&e.task_id===task&&e.instruction).map(e=>e.instruction))];
 return instructions.length?instructions.join(' / '):fallback(task,job?.options);
}
export function coverageSubset(data,task='all',origin='all'){
 return (data?.episodes||[]).filter(e=>(task==='all'||e.task_id===task)&&(origin==='all'||e.inherited===(origin==='inherited')));
}
export function coverageCounts(episodes,role){
 const counts=Array(9).fill(0);let missing=0,outside=0;
 for(const e of episodes){if(!e[`${role}_xy_mm`])missing++;else if(Number.isInteger(e[`${role}_zone`]))counts[e[`${role}_zone`]]++;else outside++;}
 return {counts,missing,outside,positioned:episodes.length-missing};
}
export function coverageRobotBases(episodes){
 const unique=new Map();for(const e of episodes){const p=e.robot_base_xy_mm;if(!Array.isArray(p)||p.length!==2||!p.every(Number.isFinite))continue;const yaw=Number.isFinite(e.robot_base_yaw_deg)?e.robot_base_yaw_deg:null,key=JSON.stringify([p,yaw]);if(!unique.has(key))unique.set(key,{xy:p,yaw});}return [...unique.values()];
}
export function coveragePlotBounds(mat,bases){
 const low=[...mat[0]],high=[...mat[1]];for(const {xy}of bases)for(let i=0;i<2;i++){low[i]=Math.min(low[i],xy[i]-35);high[i]=Math.max(high[i],xy[i]+35);}
 const side=Math.max(high[0]-low[0],high[1]-low[1]);return [low.map((v,i)=>(v+high[i]-side)/2),high.map((v,i)=>(v+low[i]+side)/2)];
}
const NS='http://www.w3.org/2000/svg';
function el(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
function svg(tag,attributes={},text){const node=document.createElementNS(NS,tag);for(const [key,value] of Object.entries(attributes))node.setAttribute(key,String(value));if(text!==undefined)node.textContent=text;return node;}
const css=`
.generation-coverage{min-width:0}.generation-coverage h3{margin:0 0 8px}.generation-coverage p{font-size:12px;line-height:1.55;color:var(--muted,#65717e)}
.generation-coverage .coverage-filters{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}.generation-coverage label{display:grid;gap:5px;font-size:12px;min-width:120px;max-width:100%}.generation-coverage select{max-width:100%;padding:7px;border:1px solid var(--border,#d5dce2);border-radius:7px;background:var(--surface,#fff);color:inherit}
.generation-coverage .coverage-tasks{display:flex;flex-wrap:wrap;gap:7px}.generation-coverage .coverage-task{font-size:12px;border:1px solid var(--border,#dce2e7);border-radius:8px;padding:6px 9px}
.generation-coverage .coverage-maps{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,285px),1fr));gap:16px}.generation-coverage figure{margin:0;min-width:0}.generation-coverage figcaption{font-size:13px;font-weight:600}.generation-coverage svg{display:block;width:100%;height:auto;max-width:570px;margin:auto;overflow:visible}.generation-coverage svg text{font:11px system-ui;fill:var(--muted,#536477)}
.generation-coverage .coverage-point{cursor:pointer;stroke:#fff;stroke-width:.7;opacity:.65;outline:none}.generation-coverage .coverage-point:hover,.generation-coverage .coverage-point:focus,.generation-coverage .coverage-point.selected{opacity:1;stroke:#13273c;stroke-width:2;r:5}.generation-coverage .coverage-zone-count{font-size:12px;font-weight:650;fill:#253d52}.generation-coverage .coverage-selection{min-height:22px;color:var(--text,#24374b)}
`;
export class GenerationCoverage{
 constructor(api,container,onSelect,taskName=id=>id.replaceAll('_',' ')){
  this.api=api;this.container=container;this.onSelect=onSelect;this.taskName=taskName;this.request=0;this.signature='';this.selected=-1;this.task='all';this.role='both';this.origin='all';
  if(container){container.classList.add('generation-coverage');const style=el('style',css);container.append(style);this.body=el('div');container.append(this.body);}
 }
 async load(job,selected=-1){
  if(!this.container)return;
  this.select(selected);
  if(!job){this.request++;this.signature='';this.data=null;this.body.replaceChildren(el('h3','Workspace coverage'),el('p','Choose a dataset to inspect its accepted episodes.'));return;}
  const signature=JSON.stringify([job.id,job.status,(job.episodes||[]).map(e=>[e.status,e.task_id])]);
  if(signature===this.signature)return;
  const previousId=this.jobId;this.signature=signature;this.jobId=job.id;const request=++this.request;
  if(previousId!==job.id){this.task='all';this.origin='all';this.data=null;}
  this.body.replaceChildren(el('h3','Workspace coverage'),el('p','Reading saved episode positions…'));
  try{
   const data=await this.api(`/api/generation/${encodeURIComponent(job.id)}/coverage`);
   if(request!==this.request)return;
   this.data=data;this.render();
  }catch(error){if(request!==this.request)return;this.signature='';this.body.replaceChildren(el('h3','Workspace coverage'),el('p',`Coverage unavailable: ${error.message||error}`));}
 }
 select(index){this.selected=Number(index);if(!this.container)return;for(const point of this.container.querySelectorAll('[data-coverage-episode]'))point.classList.toggle('selected',Number(point.dataset.coverageEpisode)===this.selected);for(const point of this.container.querySelectorAll('.coverage-point.selected'))point.parentNode.append(point);this.selectionText();}
 selectionText(){
  if(!this.selection)return;
  const item=this.data?.episodes.find(e=>e.episode_index===this.selected);
  const position=(role)=>item?.[`${role}_xy_mm`]?.map(v=>v.toFixed(1)).join(', ')??'unavailable';
  this.selection.textContent=item?`Selected episode ${item.episode_index+1} · pick (${position('source')}) mm · destination (${position('destination')}) mm`:'Click a point to inspect its episode.';
 }
 taskLabel(task){const instructions=[...new Set((this.data?.episodes||[]).filter(e=>e.task_id===task&&e.instruction&&e.instruction!==task).map(e=>e.instruction))];return instructions.length?instructions.join(' / '):this.taskName(task);}
 filter(label,value,options,changed){
  const wrap=el('label',label),select=el('select');select.setAttribute('aria-label',label);
  for(const [id,text] of options){const option=el('option',text);option.value=id;select.append(option);}select.value=value;select.addEventListener('change',()=>changed(select.value));wrap.append(select);return wrap;
 }
 render(){
  const data=this.data;if(!data)return;
  this.body.replaceChildren(el('h3','Workspace coverage'),el('p',`${data.accepted} accepted episodes · initial positions on the full calibrated mat · world X/Y in mm`));
  const tasks=el('div',undefined,'coverage-tasks');
  for(const task of data.tasks)tasks.append(el('span',`${this.taskLabel(task.task_id)}: ${task.accepted}${task.target!=null?' / '+task.target:''}`, 'coverage-task'));
  this.body.append(tasks);
  const filters=el('div',undefined,'coverage-filters');
  filters.append(this.filter('Task',this.task,[['all','All tasks'],...data.tasks.map(t=>[t.task_id,this.taskLabel(t.task_id)])],value=>{this.task=value;this.render();}),this.filter('Positions',this.role,[['both','Pick + destination'],['source','Pick only'],['destination','Destination only']],value=>{this.role=value;this.render();}),this.filter('Episodes',this.origin,[['all','All accepted'],['inherited','Inherited'],['new','New in this batch']],value=>{this.origin=value;this.render();}));
  this.body.append(filters);
  const episodes=coverageSubset(data,this.task,this.origin),maps=el('div',undefined,'coverage-maps');
  this.body.append(el('p',`${episodes.length} episodes in this view. Each zone shows its accepted position count; dots select episodes.`));
  if(data.mat_bounds_xy_mm){for(const role of this.role==='both'?['source','destination']:[this.role])maps.append(this.map(episodes,role));this.body.append(maps);}
  else this.body.append(el('p','Saved mat bounds are unavailable. Spatial zones cannot be drawn for this dataset.'));
  if(data.different_mats)this.body.append(el('p','This dataset contains different saved mat bounds. The grid uses the first recorded mat; positions outside it are reported separately.'));
  this.selection=el('p',undefined,'coverage-selection');this.body.append(this.selection);this.selectionText();
  this.body.append(el('p','Robot marker = saved base origin, not the moving arm or a physical footprint. Arrow = base local +X. Missing base metadata is not estimated.'));
  this.body.append(el('p','Pick = initial source-cube center. Destination = initial target-cube or bowl center; unstack uses the saved empty-mat placement. Inherited episodes retain their original positions.'));
 }
 map(episodes,role){
  const figure=el('figure'),title=role==='source'?'Pick positions':'Destination positions',color=role==='source'?'#257dac':'#c36d28';
  const {counts,missing,outside,positioned}=coverageCounts(episodes,role);
  figure.append(el('figcaption',title),el('p',`${positioned} saved positions · ${missing} missing${outside?' · '+outside+' outside recorded mat':''}`));
  const plot=svg('svg',{viewBox:'0 0 400 390',role:'group','aria-label':`${title}, 3 by 3 workspace zones in millimeters`}),[low,high]=this.data.mat_bounds_xy_mm;
  const bases=coverageRobotBases(episodes),[plotLow,plotHigh]=coveragePlotBounds([low,high],bases);
  const left=62,top=24,width=310,height=310,x=v=>left+(v-plotLow[0])/(plotHigh[0]-plotLow[0])*width,y=v=>top+height-(v-plotLow[1])/(plotHigh[1]-plotLow[1])*height;
  if(!bases.length)figure.append(el('p','Robot base position unavailable in saved scene metadata.'));
  const max=Math.max(1,...counts);
  for(let row=0;row<3;row++)for(let col=0;col<3;col++){
   const zone=row*3+col,px=x(low[0]+col*(high[0]-low[0])/3),py=y(low[1]+(row+1)*(high[1]-low[1])/3);
   plot.append(svg('rect',{x:px,y:py,width:(x(high[0])-x(low[0]))/3,height:(y(low[1])-y(high[1]))/3,fill:color,'fill-opacity':.035+.12*counts[zone]/max,stroke:'#bdcbd5','stroke-width':1}));
   plot.append(svg('text',{x:px+7,y:py+17,class:'coverage-zone-count'},`Z${zone+1} · ${counts[zone]}`));
  }
  for(let tick=0;tick<=3;tick++){
   const vx=low[0]+(high[0]-low[0])*tick/3,vy=low[1]+(high[1]-low[1])*tick/3;
   plot.append(svg('text',{x:x(vx),y:top+height+20,'text-anchor':'middle'},Number(vx.toFixed(1))),svg('text',{x:left-8,y:y(vy)+4,'text-anchor':'end'},Number(vy.toFixed(1))));
  }
  plot.append(svg('text',{x:left+width/2,y:382,'text-anchor':'middle'},'World X (mm)'),svg('text',{x:16,y:top+height/2,'text-anchor':'middle',transform:`rotate(-90 16 ${top+height/2})`},'World Y (mm)'));
  for(const item of episodes){
   const point=item[`${role}_xy_mm`];if(!point||item[`${role}_zone`]==null)continue;
   const circle=svg('circle',{cx:x(point[0]),cy:y(point[1]),r:3,fill:color,class:`coverage-point${item.episode_index===this.selected?' selected':''}`,tabindex:0,role:'button','aria-label':`Episode ${item.episode_index+1}, ${item.instruction||this.taskLabel(item.task_id)}, ${title}, X ${point[0].toFixed(1)}, Y ${point[1].toFixed(1)} millimeters`});
   circle.dataset.coverageEpisode=item.episode_index;circle.append(svg('title',{},`Episode ${item.episode_index+1} · ${item.inherited?'inherited':'new'} · ${item.instruction||this.taskLabel(item.task_id)}\nX ${point[0].toFixed(1)} mm · Y ${point[1].toFixed(1)} mm`));
   const activate=()=>{this.select(item.episode_index);this.onSelect(item.episode_index);};circle.addEventListener('click',activate);circle.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();activate();}});plot.append(circle);
  }
  for(const base of bases){
   const px=x(base.xy[0]),py=y(base.xy[1]),marker=svg('g',{'aria-label':`Robot base, X ${base.xy[0].toFixed(1)}, Y ${base.xy[1].toFixed(1)} millimeters`,class:'coverage-robot'});
   marker.append(svg('circle',{cx:px,cy:py,r:8,fill:'#284b39',stroke:'#fff','stroke-width':2}),svg('path',{d:`M${px-4} ${py}h8 M${px} ${py-4}v8`,stroke:'#fff','stroke-width':1.2}));
   if(base.yaw!==null){const angle=base.yaw*Math.PI/180,dx=23*Math.cos(angle),dy=-23*Math.sin(angle),ex=px+dx,ey=py+dy;marker.append(svg('path',{d:`M${px} ${py}L${ex} ${ey}M${ex-6*Math.cos(angle)-3*Math.sin(angle)} ${ey+6*Math.sin(angle)-3*Math.cos(angle)}L${ex} ${ey}L${ex-6*Math.cos(angle)+3*Math.sin(angle)} ${ey+6*Math.sin(angle)+3*Math.cos(angle)}`,stroke:'#284b39','stroke-width':2,fill:'none'}));}
   marker.append(svg('text',{x:px,y:py+23,'text-anchor':'middle'},'Robot base'),svg('title',{},`Saved base origin (${base.xy.map(v=>v.toFixed(1)).join(', ')}) mm; arrow is local +X, not arm pose.`));plot.append(marker);
  }
  // Selected points must remain visible and clickable above dense clusters.
  for(const point of [...plot.querySelectorAll('.selected')])plot.append(point);
  figure.append(plot);return figure;
 }
}
