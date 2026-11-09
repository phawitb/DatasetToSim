/** Import a real SO-101 dataset: pick library objects, then build an auto-calibrated simulation. */
const el=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;};
const pct=v=>v==null?'—':(v*100).toFixed(1)+'%';
const mm=values=>values.map(v=>Number((v*1000).toFixed(1))).join(' × ')+' mm';
const STAGES={starting:'Starting',camera:'Fitting top camera',mat:'Fitting work mat',objects:'Locating objects',replay:'Replaying episodes',complete:'Complete'};
const OUTCOMES={success:'success',not_lifted:'not lifted',no_bilateral_grasp:'no grasp',missed_bowl:'missed bowl',not_released:'not released',not_settled:'not settled',error:'error',scene_not_fitted:'scene not found'};

export class ImportView{
 constructor(root,api,toast){this.root=root;this.api=api;this.toast=toast;this.mode='list';this.items=[];this.library=[];this.selected=new Set(['eraser','green-bowl']);this.timer=null;this.current=null;this.episode=null;}
 suspend(){clearTimeout(this.timer);this.timer=null;const v=this.root.querySelector('video');if(v)v.pause();}
 async show(id){this.suspend();this.signature=null;if(id){this.mode='detail';this.currentId=id;await this.refreshDetail();}else{this.mode='list';await this.refreshList();}}
 async refreshList(){try{const data=await this.api('/api/imports');this.items=data.imports;this.busy=data.busy;}catch(e){this.toast(e.message,true);}if(this.mode==='list')this.renderList();if(this.mode==='list'&&this.items.some(i=>i.status==='running'))this.timer=setTimeout(()=>this.refreshList(),3000);}
 renderList(){
  const root=this.root;root.replaceChildren();
  const head=el('div','import-head');const text=el('div');text.append(el('span','eyebrow','IMPORTED DATASETS'),el('h2',null,'Bring in a new recording setup'),el('p','import-help','Same SO-101 robot and calibration. The top camera, work mat and object poses are fitted automatically from the images.'));
  const button=el('button','button primary','Import dataset');button.onclick=()=>this.openForm();head.append(text,button);root.append(head);
  if(!this.items.length){root.append(el('p','import-help','No imported datasets yet.'));return;}
  const list=el('div','import-list');
  for(const item of this.items){
   const card=el('button','import-card');card.type='button';
   const status=item.status==='complete'?`${item.successes}/${item.completed} replay successes`:item.status==='running'?(STAGES[item.stage]||item.stage)+'…':item.status;
   card.append(el('strong',null,item.name),el('span',null,`${item.objects.join(' · ')} · ${new Date(item.created).toLocaleString()}`),el('span','import-status '+item.status,status),el('span',null,'Camera robot-mask IoU '+pct(item.camera_mask_iou)));
   card.onclick=()=>this.show(item.id);list.append(card);
  }
  root.append(list);
 }
 async openForm(){
  this.suspend();this.mode='form';
  if(!this.library.length){try{this.library=(await this.api('/api/imports/library')).objects;}catch(e){this.toast(e.message,true);return;}}
  const root=this.root;root.replaceChildren();
  const back=el('button','text-button','← Imported datasets');back.onclick=()=>this.show();
  const form=el('form','import-form');
  const path=el('label','generation-field','Dataset folder');const input=el('input');input.required=true;input.placeholder='/Users/…/data/MY_DATASET';input.value=localStorage.getItem('import-path')||'';path.append(input,el('small',null,'LeRobot v3 dataset recorded with the SO-101 (top + wrist cameras, 30 Hz).'));
  const pick=el('fieldset','import-objects');pick.append(el('legend',null,'Objects in the images'),el('p','import-help','Choose the library models that appear in the scene: one object that is picked and the bowl it goes into.'));
  const grid=el('div','import-object-grid');
  for(const item of this.library){
   const label=el('label','import-object');const box=el('input');box.type='checkbox';box.checked=this.selected.has(item.id);box.disabled=!item.replay_supported;
   box.onchange=()=>{box.checked?this.selected.add(item.id):this.selected.delete(item.id);};
   const img=el('img');img.src=item.thumbnail_url;img.alt='';img.loading='lazy';
   const info=el('span');info.append(el('strong',null,item.name),el('small',null,(item.role==='container'?'Bowl · ':'Picked object · ')+mm(item.dimensions_m)+(item.replay_supported?'':' · replay not supported yet')));
   label.append(box,img,info);grid.append(label);
  }
  pick.append(grid);
  const status=el('p','import-help','Everything after this is computed automatically: camera, mat, object poses, then a strict replay of every episode with the original actions.');
  const submit=el('button','button primary full','Build simulation');submit.type='submit';
  form.append(path,pick,submit,status);
  form.onsubmit=async event=>{event.preventDefault();submit.disabled=true;try{localStorage.setItem('import-path',input.value.trim());const job=await this.api('/api/imports',{dataset_path:input.value.trim(),objects:[...this.selected]});await this.show(job.id);}catch(e){this.toast(e.message,true);submit.disabled=false;}};
  root.append(back,form);
 }
 async refreshDetail(){
  const id=this.currentId;
  try{this.current=await this.api('/api/imports/'+encodeURIComponent(id));}catch(e){this.toast(e.message,true);return;}
  if(this.mode!=='detail'||this.currentId!==id)return;
  const signature=JSON.stringify(this.current);if(signature!==this.signature){this.signature=signature;this.renderDetail();}
  if(this.current.status==='running')this.timer=setTimeout(()=>this.refreshDetail(),3000);
 }
 renderDetail(){
  const job=this.current,root=this.root,playing=root.querySelector('video');const keep=playing&&!playing.paused?playing.currentSrc:null;
  root.replaceChildren();
  const back=el('button','text-button','← Imported datasets');back.onclick=()=>this.show();
  const head=el('div','import-head');const title=el('div');title.append(el('span','eyebrow','IMPORTED DATASET'),el('h2',null,job.name),el('p','import-help',job.dataset_path));head.append(title);
  if(job.status==='running'){const cancel=el('button','button','Cancel');cancel.onclick=async()=>{try{await this.api(`/api/imports/${encodeURIComponent(job.id)}/cancel`,{});await this.refreshDetail();}catch(e){this.toast(e.message,true);}};head.append(cancel);}
  root.append(back,head);
  const stats=el('div','import-stats');
  const stat=(label,value,note)=>{const d=el('div');d.append(el('span',null,label),el('strong',null,value),el('small',null,note||''));stats.append(d);};
  stat('STATUS',job.status==='running'?(STAGES[job.stage]||job.stage):job.status,job.status==='running'&&job.total?`${job.completed||0} / ${job.total} episodes replayed`:(job.error||''));
  stat('REPLAY TASK SUCCESS',job.completed?`${job.successes} / ${job.completed}`:'—','Same task criteria as the shared strict replay');
  const cal=job.calibration;
  stat('CAMERA ALIGNMENT',pct(cal?.camera_report?.mean_mask_iou),'Robot orange-part mask IoU, real vs sim');
  if(cal)stat('WORK MAT',mm(cal.mat.size.slice(0,2)),`${cal.mat_report.corners_visible.filter(Boolean).length}/4 corners visible · thickness assumed 2 mm`);
  root.append(stats);
  if(job.camera_fit_image){const fig=el('figure','import-figure');const img=el('img');img.src=job.camera_fit_image+'?t='+Date.now();img.alt='Camera fit overlay';fig.append(img,el('figcaption',null,'Camera fit: green = real and sim robot agree, red = real only, blue = sim only.'));root.append(fig);}
  if(job.episodes?.length){
   const layout=el('div','import-episodes');const list=el('ol','import-episode-list');const viewer=el('div','import-viewer');
   const select=row=>{this.episode=row.episode;for(const li of list.children)li.classList.toggle('selected',Number(li.dataset.episode)===row.episode);viewer.replaceChildren();
    if(row.video){const v=el('video');v.controls=true;v.src=row.video;v.playsInline=true;viewer.append(v);}else viewer.append(el('p','import-help','Video not available.'));
    const facts=el('p','import-help',`Episode ${row.episode} · ${OUTCOMES[row.outcome]||row.outcome} · arm RMSE ${row.joint_state_rmse_native?Math.sqrt(row.joint_state_rmse_native.slice(0,5).reduce((a,x)=>a+x*x,0)/5).toFixed(2)+'°':'—'} · top robot IoU ${pct(row.top_robot_mask_iou?.mean)}`+(row.error?' · '+row.error:''));viewer.append(facts);
    if(row.scene_fit){const img=el('img','import-scene');img.src=row.scene_fit;img.alt='Fitted mat and object outlines';viewer.append(img,el('small',null,'First frame: fitted mat (magenta), picked object (green), bowl (orange).'));}};
   for(const row of job.episodes){const li=el('li');li.dataset.episode=row.episode;const b=el('button');b.type='button';b.append(el('span',null,'Episode '+String(row.episode).padStart(2,'0')),el('em','outcome '+(row.outcome==='success'?'ok':'bad'),OUTCOMES[row.outcome]||row.outcome));b.onclick=()=>select(row);li.append(b);list.append(li);}
   layout.append(list,viewer);root.append(layout);
   const chosen=job.episodes.find(r=>r.episode===this.episode)||job.episodes[0];select(chosen);
   if(keep){const v=viewer.querySelector('video');if(v&&v.src===keep)v.play().catch(()=>{});}
  }
  if(job.log?.length){const details=el('details','import-log');details.open=job.status==='running';details.append(el('summary',null,'Calibration log'),el('pre',null,job.log.join('\n')));root.append(details);}
 }
}
