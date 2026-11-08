import test from 'node:test';
import assert from 'node:assert/strict';
import {generationTaskCounts,generationTaskLabel,coverageSubset,coverageCounts} from '../web/generation-coverage.mjs';
test('accepted counts include inherited tasks outside current selection without target totals',()=>{
 const job={options:{tasks:['green']},episodes:[{status:'accepted',task_id:'red',inherited_from_job_id:'parent'},{status:'accepted',task_id:'green'},{status:'rejected',task_id:'green'}]};
 assert.deepEqual(generationTaskCounts(job),[{task_id:'green',accepted:1,target:null},{task_id:'red',accepted:1,target:null}]);
 job.options.target_counts={green:300};assert.equal(generationTaskCounts(job)[0].target,300);
});
test('coverage filters preserve journal indices and count positions, missing metadata and outliers',()=>{
 const data={episodes:[{episode_index:0,task_id:'red',inherited:true,source_xy_mm:[40,-150],source_zone:0},{episode_index:2,task_id:'green',inherited:false,source_xy_mm:null,source_zone:null},{episode_index:3,task_id:'red',inherited:false,source_xy_mm:[600,0],source_zone:null}]};
 assert.deepEqual(coverageSubset(data,'red','new').map(e=>e.episode_index),[3]);
 assert.equal(coverageSubset(data,'all','inherited').length,1);
 assert.deepEqual(coverageCounts(data.episodes,'source'),{counts:[1,0,0,0,0,0,0,0,0],missing:1,outside:1,positioned:2});
});

test('task labels use inherited recorded instructions instead of current unstack color',()=>{
 const job={options:{unstack_top:'green'},episodes:[{status:'accepted',task_id:'unstack',instruction:'Lift red off green.'}]};
 assert.equal(generationTaskLabel(job,'unstack',()=> 'Lift green off red.'),'Lift red off green.');
 job.episodes.push({status:'accepted',task_id:'unstack',instruction:'Lift green off red.'});
 assert.equal(generationTaskLabel(job,'unstack'),'Lift red off green. / Lift green off red.');
 assert.equal(generationTaskLabel(job,'empty',id=>'Pending '+id),'Pending empty');
});

test('switching datasets discards late coverage responses and unchanged polling reuses data',async()=>{
 const {GenerationCoverage}=await import('../web/generation-coverage.mjs');
 const original=globalThis.document;
 globalThis.document={createElement:()=>({append(){},replaceChildren(){}})};
 try{
  const requests=[],pending=[];
  const api=url=>{requests.push(url);return new Promise(resolve=>pending.push(resolve));};
  const host={classList:{add(){}},append(){},querySelectorAll(){return [];}};
  const viewer=new GenerationCoverage(api,host,()=>{});viewer.render=()=>{};
  const a={id:'a',status:'completed',episodes:[]},b={id:'b',status:'completed',episodes:[]};
  const first=viewer.load(a),second=viewer.load(b);
  pending[1]({job_id:'b',episodes:[]});await second;
  pending[0]({job_id:'a',episodes:[]});await first;
  assert.equal(viewer.data.job_id,'b');
  await viewer.load(b,7);assert.equal(requests.length,2);assert.equal(viewer.selected,7);
  await viewer.load(null);assert.equal(viewer.data,null);
 }finally{globalThis.document=original;}
});

test('robot origins outside the mat remain visible without changing workspace zones',async()=>{
 const {coverageRobotBases,coveragePlotBounds}=await import('../web/generation-coverage.mjs');
 const bases=coverageRobotBases([{robot_base_xy_mm:[-43,-1],robot_base_yaw_deg:0},{robot_base_xy_mm:[-43,-1],robot_base_yaw_deg:0},{robot_base_xy_mm:null}]);
 assert.equal(bases.length,1);
 const mat=[[27.5,-202.5],[432.5,202.5]],bounds=coveragePlotBounds(mat,bases);
 assert.ok(bounds[0][0]<-43&&bounds[1][0]>=432.5);
 assert.equal(bounds[1][0]-bounds[0][0],bounds[1][1]-bounds[0][1]);
 assert.deepEqual(mat,[[27.5,-202.5],[432.5,202.5]]);
});
