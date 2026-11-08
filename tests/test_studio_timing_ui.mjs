import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
function workspace(){
 const nodes=new Map();
 const element=()=>({value:'',textContent:'',children:[],disabled:false,classList:{toggle(){},add(){},remove(){}},append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;}});
 const $=selector=>{if(!nodes.has(selector))nodes.set(selector,element());return nodes.get(selector);};
 const state={page:'new',mode:'live',request:0,robot:{connected:true,torque_enabled:true},capture:{id:'capture',source:'live'},activeCapture:'capture',selection:{from:{},to:{}},plan:{motion_technique:'natural',speed_scale:1,timing_profile:'recorded',source:'live',outcome:'success',transfer_checks_passed:true,captured_at:Date.now()/1000,frames:300,duration:10,max_ik_error_mm:1}};
 const calls=[];
 const context=vm.createContext({$,state,$$:()=>[],document:{createElement:element},playback:{ready:true},selectionRoles:()=>['from','to'],updateSelection(){},pretty:x=>x||'',api:async(path,payload)=>{calls.push({path,payload});return {};},event(){},toast(){},image:async()=>null,setOutcome(){},setVideo(){},blank(){},draw(){},playbackLabel(){},updateTimeline(){},planVideo(){}});
 vm.runInContext(source.slice(source.indexOf('const motionLabels='),source.indexOf('async function poll()')),context);
 for(const [id,value]of Object.entries({'motion-technique':'natural','motion-speed':'100','motion-timing':'recorded'}))$('#'+id).value=value;
 return {$,state,calls,run:code=>vm.runInContext(code,context)};
}
test('changing only timing invalidates simulation and execution until replanning',async()=>{
 const w=workspace();w.run('updateControls()');assert.equal(w.$('#execute').disabled,false);
 w.$('#motion-timing').value='conservative';w.run('updateControls()');
 assert.equal(w.$('#simulate').disabled,true);assert.equal(w.$('#execute').disabled,true);
 await w.run("operation('execute')");assert.equal(w.calls.length,0);
});
test('plan, simulation and execution requests carry the selected matching timing profile',async()=>{
 for(const stage of ['plan','simulate','execute']){
  const w=workspace();await w.run(`operation('${stage}')`);
  assert.equal(w.calls.length,1);assert.equal(w.calls[0].payload.timing_profile,'recorded');
 }
});
test('saved plans restore recorded or legacy conservative timing and describe actual duration',async()=>{
 const w=workspace();await w.run('showPlan(state.plan)');
 assert.equal(w.$('#motion-timing').value,'recorded');
 assert.match(w.$('#plan-result').children[1].textContent,/Recorded robot pace/);
 assert.match(w.$('#motion-speed-note').textContent,/10\.0 s/);
 delete w.state.plan.timing_profile;await w.run('showPlan(state.plan)');
 assert.equal(w.$('#motion-timing').value,'conservative');
 assert.match(w.$('#motion-speed-note').textContent,/32\.2°\/s/);
});
test('timing controls cannot change during active work, executed plans or history playback',()=>{
 for(const change of [{busy:true},{selecting:true},{robot:{running:true}},{plan:{executed:true}},{page:'history'}]){
  const w=workspace();Object.assign(w.state,change);w.run('updateControls()');assert.equal(w.$('#motion-timing').disabled,true);
 }
});
