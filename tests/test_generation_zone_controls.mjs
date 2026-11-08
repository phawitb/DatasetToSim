import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultObjectZones,activeZoneObjects,zoneSelectionError,zoneSummary,ZONE_DISPLAY_ORDER,sceneObjectStates,taskObjects} from '../web/generation-zones.mjs';

test('zone display matches coverage Y-up ordering and defaults cover whole mat',()=>{
 assert.deepEqual(ZONE_DISPLAY_ORDER,[6,7,8,3,4,5,0,1,2]);
 const a=defaultObjectZones();a.red_cube.pop();assert.equal(defaultObjectZones().red_cube.length,9);
});
test('both cubes are present in bowl tasks; absent bowl and unstack release controls are contextual',()=>{
 assert.deepEqual(activeZoneObjects(['red_cube_to_green_bowl']),['red_cube','green_cube','bowl']);
 assert.deepEqual(activeZoneObjects(['green_cube_to_red_cube']),['red_cube','green_cube']);
 assert.deepEqual(activeZoneObjects(['unstack']),['red_cube','green_cube','unstack_destination']);
});
test('empty zones and disjoint unstack starts prevent submissions but normal stacking permits disjoint cells',()=>{
 const zones={...defaultObjectZones(),red_cube:[0],green_cube:[8]};
 assert.match(zoneSelectionError(zones,['unstack']),/shared zone/);
 assert.equal(zoneSelectionError(zones,['red_cube_to_green_cube']),'');
 assert.match(zoneSelectionError({...zones,red_cube:[]},['red_cube_to_green_cube']),/Red cube/);
});
test('saved settings communicate zone identities and legacy defaults',()=>{
 assert.equal(zoneSummary([0,4,8]),'Z1, Z5, Z9');
 assert.equal(zoneSummary(), 'All zones');
});

test('zone selection commits on input before the parent form refreshes controls',async()=>{
 const {ObjectZonePicker}=await import('../web/generation-zones.mjs');
 const inputs=Array.from({length:9},(_,zone)=>({dataset:{zone:String(zone)},checked:true,handlers:{},addEventListener(name,fn){this.handlers[name]=fn;}}));
 const card={dataset:{zoneObject:'red_cube'},querySelector:()=>({}),querySelectorAll:selector=>selector==='[data-zone-action]'?[]:selector==='[data-zone]:checked'?inputs.filter(n=>n.checked):inputs};
 const host={querySelectorAll:()=>[card],querySelector:()=>({})};let changed=0;
 const picker=new ObjectZonePicker(host,()=>changed++);picker.update(['red_cube_to_green_cube'],true);
 inputs[0].checked=false;inputs[0].handlers.input();
 picker.update(['red_cube_to_green_cube'],true); // enclosing form input listener
 assert.equal(inputs[0].checked,false);assert.equal(changed,1);
 assert.deepEqual(picker.settings().red_cube,[1,2,3,4,5,6,7,8]);
});

test('by default a scene holds only the task objects; others can be added',()=>{
 assert.deepEqual(taskObjects('red_cube_to_green_bowl'),['red_cube','bowl']);
 assert.deepEqual(activeZoneObjects(['red_cube_to_green_bowl'],[]),['red_cube','bowl']);
 assert.deepEqual(activeZoneObjects(['red_cube_to_green_bowl'],['green_cube']),['red_cube','green_cube','bowl']);
 assert.deepEqual(activeZoneObjects(['red_cube_to_green_cube'],[]),['red_cube','green_cube']);
 assert.deepEqual(activeZoneObjects(['red_cube_to_green_cube'],['bowl']),['red_cube','green_cube','bowl']);
 assert.deepEqual(sceneObjectStates(['red_cube_to_green_bowl','green_cube_to_green_bowl']),{red_cube:'some',green_cube:'some',bowl:'all'});
 assert.equal(zoneSelectionError({...defaultObjectZones(),green_cube:[]},['red_cube_to_green_bowl'],[]),'');
});
