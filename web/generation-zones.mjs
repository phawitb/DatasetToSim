/** Initial-center constraints use exactly the coverage map's world X/Y zones. */
export const ZONE_OBJECTS={red_cube:'Red cube',green_cube:'Green cube',bowl:'Bowl',unstack_destination:'Unstack release area'};
export const ZONE_DISPLAY_ORDER=[6,7,8,3,4,5,0,1,2];
export const defaultObjectZones=()=>Object.fromEntries(Object.keys(ZONE_OBJECTS).map(key=>[key,[0,1,2,3,4,5,6,7,8]]));
export const SCENE_OBJECTS=['red_cube','green_cube','bowl'];
// Requests saved before scene-object selection always contained both cubes.
export const LEGACY_EXTRA_OBJECTS=['red_cube','green_cube'];
/** Objects named by a task instruction (mirrors studio_generation_zones.task_objects). */
export function taskObjects(task,unstackTop='red'){
 const color=task==='unstack'?unstackTop:task.split('_')[0],other=color==='red'?'green':'red';
 return [color+'_cube',...(task.endsWith('_bowl')?['bowl']:[other+'_cube'])];
}
/** Per object: required by every selected task, by some of them, or by none. */
export function sceneObjectStates(tasks,unstackTop='red'){
 return Object.fromEntries(SCENE_OBJECTS.map(key=>{const n=tasks.filter(t=>taskObjects(t,unstackTop).includes(key)).length;return [key,!tasks.length?'none':n===tasks.length?'all':n?'some':'none'];}));
}
export function activeZoneObjects(tasks,extras=LEGACY_EXTRA_OBJECTS,unstackTop='red'){
 const present=new Set([...tasks.flatMap(t=>taskObjects(t,unstackTop)),...extras]);
 return [...SCENE_OBJECTS.filter(key=>present.has(key)),...(tasks.includes('unstack')?['unstack_destination']:[])];
}
export function zoneSelectionError(zones,tasks,extras=LEGACY_EXTRA_OBJECTS){
 for(const key of activeZoneObjects(tasks,extras))if(!zones[key]?.length)return `Select at least one zone for ${ZONE_OBJECTS[key]}.`;
 if(tasks.includes('unstack')&&!zones.red_cube.some(z=>zones.green_cube.includes(z)))return 'Unstack needs at least one shared zone for both cubes to start stacked.';
 return '';
}
export const zoneSummary=zones=>!zones||zones.length===9?'All zones':zones.map(z=>`Z${z+1}`).join(', ');
export class ObjectZonePicker{
 constructor(host,onChange){
  this.host=host;this.values=defaultObjectZones();this.extraObjects=[];this.tasks=[];this.enabled=true;this.onChange=onChange;
  host.innerHTML=`<div class="generation-card-heading"><h3>Object placement zones</h3><span>Initial positions · 3 × 3 mat</span></div>
   <fieldset class="scene-objects"><legend>Objects in scene</legend><p class="generation-help">By default a scene contains only the objects named in the task instruction. Add others as distractors or obstacles.</p><div class="scene-object-options">${SCENE_OBJECTS.map(key=>`<label class="scene-object"><input type="checkbox" data-scene-object="${key}"><span class="object-zone-dot ${key}"></span><span>${ZONE_OBJECTS[key]}</span><small></small></label>`).join('')}</div></fieldset>
   <p class="generation-help">Select where each object's center may start. Zones match Workspace coverage. Final positions follow the task; these are not arm-motion limits. Objects must still fit on the mat and pass simulation checks.</p>
   <div class="object-zone-cards">${Object.entries(ZONE_OBJECTS).map(([key,label])=>`<fieldset class="object-zone-card" data-zone-object="${key}"><legend><span class="object-zone-dot ${key}"></span>${label}</legend><div class="object-zone-orientation">World +Y ↑</div><div class="object-zone-map"><span class="object-zone-robot" title="Robot is on the negative-X side of the calibrated mat">Robot →</span><div class="object-zone-grid">${ZONE_DISPLAY_ORDER.map(z=>`<label><input type="checkbox" data-zone="${z}" aria-label="${label} Z${z+1}" checked><span>Z${z+1}</span></label>`).join('')}</div></div><div class="object-zone-orientation">World +X →</div><div class="object-zone-actions"><button type="button" data-zone-action="all">All zones</button><button type="button" data-zone-action="clear">Clear</button></div><small class="object-zone-selection"></small></fieldset>`).join('')}</div>
   <p class="generation-help object-zone-unstack" hidden>Unstack: both cubes start in a shared allowed zone. The release area specifies where the top cube is placed on the empty mat.</p>
   <p class="generation-help object-zone-disabled" hidden>Fixed reference layout is selected. Enable Randomize object layout to apply these zone choices.</p><p class="object-zone-error" role="status"></p>`;
  for(const input of host.querySelectorAll('[data-scene-object]'))if(input.dataset?.sceneObject)input.addEventListener('input',()=>{
   const key=input.dataset.sceneObject;this.extraObjects=input.checked?[...new Set([...this.extraObjects,key])]:this.extraObjects.filter(k=>k!==key);
   this.update(this.tasks,this.enabled);onChange();
  });
  for(const card of host.querySelectorAll('[data-zone-object]')){
   const key=card.dataset.zoneObject;
   // Commit before the enclosing form's bubbling input handler refreshes controls.
   for(const input of card.querySelectorAll('[data-zone]'))input.addEventListener('input',()=>{
    this.values[key]=[...card.querySelectorAll('[data-zone]:checked')].map(n=>Number(n.dataset.zone)).sort((a,b)=>a-b);
    this.update(this.tasks,this.enabled);onChange();
   });
   for(const button of card.querySelectorAll('[data-zone-action]'))button.addEventListener('click',()=>{
    this.values[key]=button.dataset.zoneAction==='all'?[0,1,2,3,4,5,6,7,8]:[];this.update(this.tasks,this.enabled);onChange();
   });
  }
 }
 restore(zones={},extras=LEGACY_EXTRA_OBJECTS){this.values=defaultObjectZones();for(const key of Object.keys(this.values))if(Array.isArray(zones[key]))this.values[key]=[...zones[key]];this.extraObjects=[...extras];this.update(this.tasks,this.enabled);}
 settings(){const result=defaultObjectZones();if(this.enabled)for(const key of activeZoneObjects(this.tasks,this.extras()))result[key]=[...this.values[key]];return result;}
 /** Optional objects to add; objects every selected task already needs are omitted. */
 extras(){const states=sceneObjectStates(this.tasks);return SCENE_OBJECTS.filter(key=>this.extraObjects.includes(key)&&states[key]!=='all');}
 update(tasks,enabled){
  this.tasks=tasks;this.enabled=enabled;const states=sceneObjectStates(tasks),active=activeZoneObjects(tasks,this.extras());
  for(const input of this.host.querySelectorAll('[data-scene-object]')){
   const key=input.dataset?.sceneObject;if(!key)continue;const state=states[key];
   input.checked=state==='all'||this.extraObjects.includes(key);input.disabled=state==='all';
   const note=input.closest?.('label')?.querySelector('small');if(note)note.textContent=state==='all'?'Required by task':state==='some'?'Required by some tasks · add to the others':'Optional';
  }
  for(const card of this.host.querySelectorAll('[data-zone-object]')){
   const key=card.dataset.zoneObject;card.hidden=!active.includes(key);card.disabled=!enabled;
   for(const input of card.querySelectorAll('[data-zone]'))input.checked=this.values[key].includes(Number(input.dataset.zone));
   card.querySelector('.object-zone-selection').textContent=this.values[key].length?zoneSummary(this.values[key]):'No zones selected';
  }
  this.host.querySelector('.object-zone-unstack').hidden=!tasks.includes('unstack');
  this.host.querySelector('.object-zone-disabled').hidden=enabled;
  this.error=enabled?zoneSelectionError(this.values,tasks,this.extras()):'';this.host.querySelector('.object-zone-error').textContent=this.error;
 }
}
