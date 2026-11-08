import test from 'node:test';
import assert from 'node:assert/strict';
import {replaySettings} from '../web/replay-builder.mjs';
test('real replay appearance settings cannot inject layout, physics or motion changes',()=>{
 const result=replaySettings(id=>({checked:id==='real-background',value:id==='background'?'calibrated-real':'1'}));
 assert.equal(result.real_background,true);
 assert.equal(result.background_id,'calibrated-real');
 assert.equal(result.light_intensity,1);
 for(const key of ['randomize_layout','physics_randomization','speed_scale','motion_technique'])assert.equal(key in result,false);
});
import {ReplayDatasetBuilder} from '../web/replay-builder.mjs';
function deferred(){let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};}
function fake(){
 const nodes=new Map();const node=()=>({textContent:'old',value:'',classList:{remove(){},add(){}},replaceChildren(){this.textContent='';}});
 const builder=Object.create(ReplayDatasetBuilder.prototype);Object.assign(builder,{version:0,previewVersion:0,host:{classList:{remove(){},add(){}},querySelector:()=>({reportValidity:()=>true})},q:id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);},update(){},renderBackgrounds(){},changed(){},ids:()=>[1],payload:()=>({episode_ids:[1]})});return builder;
}
test('late source responses cannot replace the most recently opened dataset',async()=>{
 const builder=fake(),a=deferred(),b=deferred();let calls=0;
 builder.api=path=>path.includes('assets')?Promise.resolve({backgrounds:[]}):(++calls===1?a.promise:b.promise);
 const first=builder.open('first'),second=builder.open('second');
 assert.equal(builder.q('preview-images').textContent,'');
 b.resolve({sources:[{id:'second',name:'Second',episode_count:0,successes:0,failures:0,episodes:[]}]});await second;
 a.resolve({sources:[{id:'first',name:'First',episode_count:0,successes:0,failures:0,episodes:[]}]});await first;
 assert.equal(builder.source.id,'second');
});
test('a submitted job may finish after navigation without pulling the user away',async()=>{
 const builder=fake(),response=deferred();builder.api=()=>response.promise;let opened=0;builder.onBuilt=()=>opened++;
 const submitted=builder.submit();builder.suspend();response.resolve({id:'job',status:'queued'});await submitted;
 assert.equal(opened,0);assert.equal(builder.pending,false);
});
