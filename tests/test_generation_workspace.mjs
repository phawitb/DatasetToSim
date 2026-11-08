import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {generationTaskCounts,generationTaskLabel} from '../web/generation-coverage.mjs';
import {ObjectZonePicker,ZONE_OBJECTS,activeZoneObjects,zoneSummary} from '../web/generation-zones.mjs';

// Exercise the generation workspace without starting a server or robot session.
const source=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
function workspace(){
 const nodes=new Map();
 const element=()=>({value:'',textContent:'',dataset:{},children:[],checked:false,
  classList:{toggle(){},remove(){},add(){}},setAttribute(){},removeAttribute(){},querySelectorAll:()=>[],querySelector:()=>({}),
  append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;},reportValidity:()=>true});
 const $=selector=>{if(!nodes.has(selector))nodes.set(selector,element());return nodes.get(selector);};
 const task=Object.assign(element(),{value:'unstack',checked:true,nextElementSibling:element(),closest:()=>element()});
 const $$=selector=>selector.includes('generation-task')?[task]:selector==='#generation-episode-list button'?$('#generation-episode-list').children:[];
 const context=vm.createContext({$,$$,document:{createElement:element},GenerationVisualizer:class{load(){}suspend(){}},GenerationCoverage:class{load(){}select(){}},generationTaskCounts,generationTaskLabel,ObjectZonePicker,ZONE_OBJECTS,activeZoneObjects,zoneSummary,
  state:{page:'generate',restarting:false},pretty:value=>value||'',motionLabels:{natural:'Natural'},
  history:{replaceState(){}},URLSearchParams,Option:class{constructor(text,value){this.text=text;this.value=value;}},
  api:async()=>({}),toast(){},setTimeout,clearTimeout});
 vm.runInContext(source.slice(source.indexOf('const generationViewer='),source.indexOf('(async()=>{try{const data=await api(\'/api/bootstrap\')')),context);
 for(const [id,value]of Object.entries({name:'test-set',count:'1',seed:'42',motion:'natural',timing:'recorded',speed:'1',validation:'0.2',unstack:'red'}))$('#generation-'+id).value=value;
 return {$,$$,task,run:code=>vm.runInContext(code,context),context};
}
test('the chosen upper cube changes the full build instruction, independently of a saved job',()=>{
 const w=workspace();w.$('#generation-unstack').value='green';
 w.run("generation.loaded=true;generation.job={id:'saved',options:{unstack_top:'red'}};updateGenerationForm()");
 assert.equal(w.task.nextElementSibling.textContent,'Lift the green cube off the red cube and place it on the empty mat.');
 w.run("generation.job.episodes=[{status:'accepted',task_id:'unstack'}];renderGenerationEpisodes()");
 assert.match(w.$('#generation-episode-list').children[0].textContent,/Lift the red cube off the green cube and place it on the empty mat\./);
});
test('successful submission opens Visualize and retains the returned dataset',async()=>{
 const w=workspace();
 w.run("generation.loaded=true;refreshGeneration=async()=>{};api=async(path,payload)=>({id:'new-job',name:'test-set',status:'queued',options:payload,progress:{requested:1},episodes:[]})");
 await w.$('#generation-form').onsubmit({preventDefault(){}});
 assert.equal(w.run('generation.tab'),'visualize');
 assert.equal(w.run('generation.job.id'),'new-job');
 assert.equal(w.run('generation.job.options.timing_profile'),'recorded');
});
test('new attempts from polling preserve the selected accepted episode',()=>{
 const w=workspace();
 w.run("generation.job={id:'saved',options:{unstack_top:'green'},episodes:[{status:'accepted',task_id:'unstack'},{status:'accepted',task_id:'red_cube_to_green_bowl'}]};generation.episodeIndex=1;renderGenerationEpisodes()");
 w.run("generation.job.episodes.push({status:'rejected',task_id:'unstack'});renderGenerationEpisodes()");
 assert.equal(w.run('generation.episodeIndex'),1);
 assert.equal(w.$('#generation-episode-list').children.length,2);
 assert.match(w.$('#generation-episode-list').children[1].textContent,/Put the red cube in the green bowl\./);
});

test('dataset information shows measured duration and identifies legacy timing correctly',()=>{
 const w=workspace();
 w.run("generation.job={id:'old',created_at:1735689600,options:{tasks:['unstack']},summary:{fps:30,total_frames:360,total_duration_s:12,cameras:['top','wrist'],resolution:[640,480],format:'LeRobot v3'}};renderGenerationInfo()");
 const info=Object.fromEntries(w.$('#generation-dataset-info').children.map(item=>item.children.map(node=>node.textContent)));
 assert.equal(info['Frames / duration'],'360 frames · 12.0 s');
 assert.equal(info['Motion timing'],'Conservative studio pace');
 assert.equal(info['Tasks'],'Lift the red cube off the green cube and place it on the empty mat.');
});

test('real-image options submit independently while preserving existing variation settings',async()=>{
 for(const [background,processing] of [[true,false],[false,true],[true,true],[false,false]]){
  const w=workspace();
  w.$('#generation-real-background').checked=background;
  w.$('#generation-real-processing').checked=processing;
  w.$('#generation-layout').checked=true;
  w.$('#generation-sensor').checked=true;
  w.$('#generation-visual').checked=false;
  w.$('#generation-camera').checked=false;
  w.$('#generation-physics').checked=false;
  w.run("generation.loaded=true;refreshGeneration=async()=>{};api=async(path,payload)=>({id:'new-job',status:'queued',options:payload,episodes:[]})");
  await w.$('#generation-form').onsubmit({preventDefault(){}});
  const options=w.run('generation.job.options');
  assert.equal(options.real_background,background);
  assert.equal(options.real_image_processing,processing);
  assert.equal(options.randomize_layout,true);
  assert.equal(options.sensor_augmentation,true);
  assert.equal(options.visual_randomization,false);
  assert.equal(options.camera_jitter,false);
  assert.equal(options.physics_randomization,false);
 }
});
test('saved settings distinguish real-image processing from sensor augmentation and legacy jobs',()=>{
 const w=workspace();
 const variations=()=>{const values=w.$('#generation-batch-values').children;return values[values.findIndex(node=>node.textContent==='Enabled variations')+1].textContent;};
 w.run("generation.job={id:'new',options:{tasks:['red_cube_to_green_cube'],randomize_layout:true,real_background:true,real_image_processing:true,sensor_augmentation:false,visual_randomization:false,camera_jitter:false,physics_randomization:false}};renderGenerationBatchSettings()");
 assert.equal(variations(),'Object layout · Real-image background · Match real camera appearance');
 w.run("generation.job={id:'legacy',options:{tasks:['red_cube_to_green_cube'],sensor_augmentation:true}};renderGenerationBatchSettings()");
 assert.equal(variations(),'Sensor augmentation');
});

test('Add episodes restores saved settings, advances the seed, and leaves the parent untouched',()=>{
 const w=workspace();
 w.run(`generation.loaded=true;generation.job={id:'parent',name:'original',status:'completed',next_seed:81,export:{valid:true,total_episodes:7},options:{tasks:['unstack'],episodes_per_task:3,seed:42,motion_technique:'flow',timing_profile:'recorded',speed_scale:.75,randomize_layout:false,real_background:true,real_image_processing:true,visual_randomization:false,camera_jitter:true,sensor_augmentation:false,physics_randomization:true,validation_fraction:.3,unstack_top:'green',background_id:'uploaded-1',lighting_randomization:true,light_intensity:1.7,light_direction:-80,light_elevation:30,light_spread:60,shadow_strength:.8}}`);
 const original=w.run('JSON.stringify(generation.job)');
 w.$('#generation-add-episodes').onclick();
 assert.equal(w.run('generation.tab'),'build');
 const settings=w.run('collectGenerationSettings()');
 assert.equal(settings.parent_job_id,'parent');assert.equal(settings.seed,81);
 assert.equal(settings.episodes_per_task,3);assert.equal(settings.motion_technique,'flow');
 assert.equal(settings.speed_scale,.75);assert.equal(settings.randomize_layout,false);
 assert.equal(settings.background_id,'uploaded-1');assert.equal(settings.lighting_randomization,true);
 assert.equal(settings.light_direction,-80);assert.equal(settings.light_intensity,1.7);
 assert.equal(settings.unstack_top,'green');assert.equal(settings.real_image_processing,true);
 assert.match(w.$('#generation-extension-summary').textContent,/7 existing.*3 new.*10/);
 w.$('#generation-count').value='5';w.$('#generation-form').oninput();
 assert.match(w.$('#generation-extension-summary').textContent,/7 existing.*5 new.*12/);
 assert.equal(w.run('JSON.stringify(generation.job)'),original);
 w.$('#generation-new-dataset').onclick();
 assert.equal(w.run('collectGenerationSettings().parent_job_id'),undefined);
});

test('preview uses the current generation payload and renders five labelled camera pairs',async()=>{
 const w=workspace();let previewPayload,submittedPayload;
 w.run('generation.loaded=true');
 w.$('#generation-background').value='uploaded-2';w.$('#generation-light-direction').value='-25';
 w.$('#generation-light-intensity').value='1.4';w.$('#generation-light-elevation').value='45';
 w.$('#generation-light-spread').value='20';w.$('#generation-shadow-strength').value='.7';
 w.$('#generation-lighting').checked=true;
 w.context.api=async(path,payload)=>{
  if(path==='/api/generation-preview'){previewPayload=JSON.parse(JSON.stringify(payload));return{id:'preview-1',status:'completed',settings_hash:'hash',images:Array.from({length:5},(_,i)=>({index:i,task_id:'unstack',seed:42+i,top_url:`/top-${i}.png`,wrist_url:`/wrist-${i}.png`}))};}
  if(path==='/api/generation'){submittedPayload=JSON.parse(JSON.stringify(payload));return{id:'next',status:'queued',options:payload,episodes:[]};}
 };
 await w.$('#generation-preview-button').onclick();
 const rows=w.$('#generation-still-results').children;
 assert.equal(rows.length,5);assert.match(rows[0].children[0].textContent,/Lift the red cube/);
 assert.equal(rows[0].children[1].children.length,2);
 assert.equal(previewPayload.background_id,'uploaded-2');assert.equal(previewPayload.light_direction,-25);
 assert.equal(previewPayload.lighting_randomization,true);assert.equal(previewPayload.light_intensity,1.4);
 w.run('refreshGeneration=async()=>{}');
 await w.$('#generation-form').onsubmit({preventDefault(){}});
 assert.deepEqual(submittedPayload,previewPayload);
});

test('a settings change discards late preview responses',async()=>{
 const w=workspace();w.run('generation.loaded=true');
 let resolve;w.context.api=()=>new Promise(done=>{resolve=done;});
 const pending=w.$('#generation-preview-button').onclick();
 w.$('#generation-seed').value='99';w.$('#generation-form').oninput();
 resolve({id:'old',status:'completed',images:[{index:0,task_id:'unstack',seed:42,top_url:'/old.png',wrist_url:'/old-wrist.png'}]});
 await pending;
 assert.equal(w.$('#generation-still-results').children.length,0);
 assert.match(w.$('#generation-still-status').textContent,/changed/i);
 assert.equal(w.$('#generation-preview-button').disabled,false);
});

test('background upload blocks generation until the selected asset is available',async()=>{
 const w=workspace();w.run('generation.loaded=true');
 let resolve,upload;
 w.context.api=(path,body,options)=>{upload={path,body,options};return new Promise(done=>{resolve=done;});};
 const file={name:'wood mat.png',type:'image/png',size:10};w.$('#generation-background-upload').files=[file];
 const pending=w.$('#generation-background-upload').onchange();
 assert.equal(w.$('#generation-submit').disabled,true);assert.equal(w.$('#generation-preview-button').disabled,true);
 resolve({background:{id:'wood',name:'wood mat',url:'/wood.png',mode:'uploaded',sha256:'abc'}});await pending;
 assert.equal(upload.path,'/api/generation-assets/upload');assert.equal(upload.body,file);
 assert.equal(upload.options.headers['X-Filename'],'wood%20mat.png');
 assert.equal(w.run('collectGenerationSettings().background_id'),'wood');
 assert.equal(w.$('#generation-real-background').checked,true);
 assert.match(w.$('#generation-background-note').textContent,/uncalibrated/i);
 assert.equal(w.$('#generation-submit').disabled,false);
});

test('preview polling completes asynchronously and stops after a bounded wait',async()=>{
 for(const finish of [true,false]){
  const w=workspace();w.run('generation.loaded=true');let scheduled=null,requests=0;
  w.context.setTimeout=fn=>{scheduled=fn;return 1;};w.context.clearTimeout=()=>{scheduled=null;};
  w.context.api=async path=>{
   if(path==='/api/generation-preview')return{id:'p',status:'queued',images:[]};
   assert.equal(path,'/api/generation-preview/p');requests++;
   return finish?{id:'p',status:'completed',images:[{index:0,task_id:'unstack',seed:42,top_url:'/top.png',wrist_url:'/wrist.png'}],settings_hash:'done'}:{id:'p',status:'running',images:[]};
  };
  await w.$('#generation-preview-button').onclick();
  assert.equal(w.$('#generation-submit').disabled,true);
  for(let i=0;scheduled&&i<130;i++){const poll=scheduled;scheduled=null;await poll();}
  assert.equal(scheduled,null);assert.equal(w.$('#generation-submit').disabled,false);
  if(finish){assert.equal(requests,1);assert.equal(w.$('#generation-still-results').dataset.settingsHash,'done');}
  else{assert.ok(requests<=120);assert.match(w.$('#generation-still-status').textContent,/taking too long/);}
 }
});

test('a pending preview poll cannot repopulate images after settings change',async()=>{
 const w=workspace();w.run('generation.loaded=true');let scheduled,resolve;
 w.context.setTimeout=fn=>{scheduled=fn;return 1;};w.context.clearTimeout=()=>{scheduled=null;};
 w.context.api=async path=>path==='/api/generation-preview'?{id:'old',status:'queued'}:await new Promise(done=>{resolve=done;});
 await w.$('#generation-preview-button').onclick();const poll=scheduled();
 w.$('#generation-light-direction').value='90';w.$('#generation-form').oninput();
 resolve({id:'old',status:'completed',images:[{index:0,task_id:'unstack',seed:42,top_url:'/old.png',wrist_url:'/old.png'}]});await poll;
 assert.equal(w.$('#generation-still-results').children.length,0);assert.equal(scheduled,null);
 assert.equal(w.$('#generation-submit').disabled,false);
});

test('preview and upload failures leave the form usable and show an actionable error',async()=>{
 const w=workspace();w.run('generation.loaded=true');
 w.context.api=async()=>{throw Error('Another graphics job is active. Try again shortly.');};
 await w.$('#generation-preview-button').onclick();
 assert.match(w.$('#generation-still-status').textContent,/Another graphics job/);
 assert.equal(w.$('#generation-preview-button').disabled,false);
 w.$('#generation-background').value='calibrated-real';
 w.$('#generation-background-upload').files=[{name:'image.webp',type:'image/webp',size:10}];
 await w.$('#generation-background-upload').onchange();
 assert.match(w.$('#generation-background-status').textContent,/Upload failed/);
 assert.equal(w.run('collectGenerationSettings().background_id'),'calibrated-real');
 assert.equal(w.$('#generation-submit').disabled,false);
});

test('raw background uploads retain the studio authorization headers and original file bytes',async()=>{
 const requests=[],context=vm.createContext({clientId:'client-1',state:{token:'token-1'},fetch:async(path,options)=>{requests.push({path,options});return{ok:true,json:async()=>({background:{id:'saved'}})};}});
 vm.runInContext(source.slice(source.indexOf('async function api('),source.indexOf('\nfunction image(')),context);
 const file={type:'image/png',bytes:'unchanged'};context.file=file;
 await vm.runInContext("api('/api/generation-assets/upload',file,{headers:{'X-Filename':'mat.png'}})",context);
 const {options}=requests[0];assert.equal(options.body,file);
 assert.equal(options.headers['X-Studio-Token'],'token-1');assert.equal(options.headers['X-Studio-Client'],'client-1');
 assert.equal(options.headers['Content-Type'],'image/png');assert.equal(options.headers['X-Filename'],'mat.png');
 await vm.runInContext("api('/api/generation',{name:'test'})",context);
 assert.equal(requests[1].options.body,'{"name":"test"}');assert.equal(requests[1].options.headers['Content-Type'],'application/json');
});

test('combined datasets retain inherited episode instructions and show lineage counts',()=>{
 const w=workspace();
 w.run(`generation.job={id:'revision',parent_job_id:'parent',status:'completed',lineage:{inherited_count:7},progress:{inherited_count:7},options:{tasks:['unstack'],unstack_top:'green'},episodes:[{status:'accepted',task_id:'unstack',instruction:'Lift the red cube off the green cube and place it on the empty mat.'},{status:'accepted',task_id:'unstack',instruction:'Lift the green cube off the red cube and place it on the empty mat.'}]};renderGenerationEpisodes();renderGenerationInfo();renderGenerationBatchSettings()`);
 const buttons=w.$('#generation-episode-list').children;
 assert.match(buttons[0].textContent,/Lift the red cube off the green cube/);
 assert.match(buttons[1].textContent,/Lift the green cube off the red cube/);
 const info=Object.fromEntries(w.$('#generation-dataset-info').children.map(item=>item.children.map(node=>node.textContent)));
 assert.equal(info['Extended from'],'parent · 7 inherited episodes');
 assert.match(info.Tasks,/Lift the red cube off the green cube/);assert.match(info.Tasks,/Lift the green cube off the red cube/);
 const values=w.$('#generation-batch-values').children;
 assert.equal(values[values.findIndex(node=>node.textContent==='Tasks')+1].textContent,'Lift the green cube off the red cube and place it on the empty mat.');
});

test('active extensions summarize inherited and requested tasks, completed datasets only accepted tasks',()=>{
 const w=workspace();
 w.run(`generation.job={id:'revision',parent_job_id:'parent',status:'running',progress:{inherited_count:2},options:{tasks:['green_cube_to_green_bowl']},episodes:[{status:'accepted',task_id:'red_cube_to_green_cube',instruction:'Stack the red cube on the green cube.'},{status:'rejected',task_id:'green_cube_to_red_cube',instruction:'Stack the green cube on the red cube.'}]};renderGenerationInfo()`);
 const info=()=>Object.fromEntries(w.$('#generation-dataset-info').children.map(item=>item.children.map(node=>node.textContent)));
 assert.equal(info()['Extended from'],'parent · 2 inherited episodes');
 assert.equal(info().Tasks,'Stack the red cube on the green cube. Put the green cube in the green bowl.');
 w.run(`generation.job.status='completed';renderGenerationInfo()`);
 assert.equal(info().Tasks,'Stack the red cube on the green cube.');
});

test('total-target jobs show accepted counts per task, excluding rejects',()=>{
 const w=workspace();
 w.run(`generation.job={id:'balanced',options:{tasks:['red_cube_to_green_cube','green_cube_to_red_cube'],target_counts:{red_cube_to_green_cube:300,green_cube_to_red_cube:300}},episodes:[{status:'accepted',task_id:'red_cube_to_green_cube'},{status:'rejected',task_id:'red_cube_to_green_cube'}]};renderGenerationInfo()`);
 const info=Object.fromEntries(w.$('#generation-dataset-info').children.map(item=>item.children.map(node=>node.textContent)));
 assert.equal(info['Stack the red cube on the green cube.'],'1 / 300 accepted episodes');
 assert.equal(info['Stack the green cube on the red cube.'],'0 / 300 accepted episodes');
});


test('delete requires selected-job confirmation and advances to remaining dataset',async()=>{
 const w=workspace();w.run("generation.loaded=true;generation.job={id:'remove',name:'Old set',status:'completed',episodes:[]};generation.jobs=[{id:'keep'}]");
 const calls=[];w.context.api=async path=>{calls.push(path);return {deleted:true};};
 w.run("refreshGeneration=async()=>{generation.jobs=[{id:'keep'}]};loadGenerationJob=async id=>{generation.job={id,status:'completed',episodes:[]}};");
 w.$('#generation-delete').onclick();assert.equal(w.run('generation.deleteId'),'remove');
 await w.$('#generation-delete-confirm').onclick();assert.deepEqual(calls,['/api/generation/remove/delete']);assert.equal(w.run('generation.job.id'),'keep');
});

test('stale deletion prompt never deletes a different selected dataset',async()=>{
 const w=workspace();w.run("generation.job={id:'first',name:'First',status:'completed'}");w.$('#generation-delete').onclick();
 w.run("generation.job={id:'second',name:'Second',status:'completed'}");
 let called=false;w.context.api=async()=>{called=true;};await w.$('#generation-delete-confirm').onclick();assert.equal(called,false);
});

test('object zones round-trip through Add episodes settings, preview payload and saved summary',()=>{
 const w=workspace();w.$('#generation-layout').checked=true;
 w.run("restoreGenerationSettings({tasks:['unstack'],randomize_layout:true,object_zones:{red_cube:[0,4],green_cube:[4,8],unstack_destination:[2]}});updateGenerationForm()");
 const settings=w.run('collectGenerationSettings()');
 assert.deepEqual(settings.object_zones.red_cube,[0,4]);
 assert.deepEqual(settings.object_zones.green_cube,[4,8]);
 assert.deepEqual(settings.object_zones.unstack_destination,[2]);
 w.run('generation.job={options:collectGenerationSettings()};renderGenerationBatchSettings()');
 assert.ok(w.$('#generation-batch-values').children.some(n=>n.textContent==='Z1, Z5'));
});

test('invalid zone selection blocks both preview and submit; fixed layout emits unrestricted zones',async()=>{
 const w=workspace();let calls=0;w.context.api=async()=>{calls++;return {};};
 w.run("generation.loaded=true;restoreGenerationSettings({tasks:['unstack'],randomize_layout:true,object_zones:{red_cube:[0],green_cube:[8]}});updateGenerationForm()");
 assert.equal(w.$('#generation-submit').disabled,true);
 assert.equal(w.$('#generation-preview-button').disabled,true);
 await w.run('startGenerationStills()');await w.$('#generation-form').onsubmit({preventDefault(){}});
 assert.equal(calls,0);
 w.$('#generation-layout').checked=false;
 const settings=w.run('collectGenerationSettings()');assert.equal(settings.object_zones.red_cube.length,9);
 w.$('#generation-layout').checked=true;assert.deepEqual(w.run('collectGenerationSettings().object_zones.red_cube'),[0]);
});
