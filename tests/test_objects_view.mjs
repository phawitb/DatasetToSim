import test from 'node:test';
import assert from 'node:assert/strict';
import {triangleBuffer, formatDimensions, filterObjects} from '../web/objects-view.mjs';

test('preview preserves real source vertices and computes finite triangle normals',()=>{
 const result=triangleBuffer({vertices:[0,0,0,1,0,0,0,1,0],faces:[0,1,2]},2);
 assert.equal(result.length,18);
 assert.deepEqual([...result.slice(0,6)],[0,0,0,0,0,1]);
 assert.deepEqual([...result.slice(6,9)],[.5,0,0]);
 assert.ok([...result].every(Number.isFinite));
});
test('catalog search includes actual task instructions',()=>{
 const items=[{name:'Green bowl',category:'Task object',tasks:[{instruction:'Put a cube in the bowl.'}]},{name:'Table',category:'Environment',tasks:[]}];
 assert.equal(filterObjects(items,'cube').length,1);
 assert.equal(filterObjects(items,'environment')[0].name,'Table');
 assert.equal(formatDimensions([.0195,.0195,.0195]),'19.5 × 19.5 × 19.5 mm');
});

test('model search finds color variants and swatches retain simulation RGB',async()=>{
 const {materialColor}=await import('../web/objects-view.mjs');
 const items=[{name:'Cube',category:'Task object',tasks:[],variants:[{name:'Red cube',label:'Red'},{name:'Green cube',label:'Green'}]}];
 assert.equal(filterObjects(items,'green')[0].name,'Cube');
 assert.equal(materialColor([.85,.045,.035,1]),'rgb(217,11,9)');
});

test('multipart appearance tiles retain each original material without recoloring',async()=>{
 const {paletteBackground}=await import('../web/objects-view.mjs');
 assert.equal(paletteBackground([[.5,.65,.6,1]]),'rgb(128,166,153)');
 const palette=paletteBackground([[1,.38,.055,1],[.1,.1,.1,1]]);
 assert.ok(palette.includes('rgb(255,97,14) 0% 50%'));
 assert.ok(palette.includes('rgb(26,26,26) 50% 100%'));
});

function fakeNode(tag='div'){
 return {tag,style:{},dataset:{},attrs:{},children:[],value:'',setAttribute(key,value){this.attrs[key]=value;},append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;}};
}
test('square color choices select exact variants while lighting only changes shading',async()=>{
 const {ObjectsView}=await import('../web/objects-view.mjs');
 const previous=globalThis.document;globalThis.document={createElement:fakeNode};
 try{
  const nodes=new Map(),view=Object.create(ObjectsView.prototype),selected=[],lighting=[];
  view.find=selector=>{if(!nodes.has(selector))nodes.set(selector,fakeNode());return nodes.get(selector);};
  view.lighting='material';view.selectVariant=id=>selected.push(id);view.preview={setLighting:value=>lighting.push(value)};
  const sage={id:'sage',label:'Sage green',rgba:[.5,.65,.6,1],materials:[[.5,.65,.6,1]],sources:[{label:'Real dataset Simulation',path:'archived/scene.xml'}]};
  const green={...sage,id:'green',label:'Green',materials:[[.045,.65,.09,1]]},model={variants:[sage,green]};
  view.renderControls(model,sage);
  const choices=nodes.get('#objects-variant').children;
  assert.deepEqual(choices.map(choice=>choice.tag),['button','button']);
  assert.deepEqual(choices.map(choice=>choice.attrs['aria-pressed']),['true','false']);
  choices[1].onclick();assert.deepEqual(selected,['green']);
  const lit=nodes.get('#objects-lighting').children[1].children[0];lit.onclick();
  assert.deepEqual(lighting,[true]);assert.deepEqual(selected,['green']);
  assert.equal(view.lighting,'studio');
  assert.equal(nodes.get('#objects-lighting').children[1].children[0].attrs['aria-pressed'],'true');
 }finally{globalThis.document=previous;}
});
test('model cards use thumbnail images without requesting full meshes',async()=>{
 const {ObjectsView}=await import('../web/objects-view.mjs');
 const previous=globalThis.document;globalThis.document={createElement:fakeNode};
 try{
  const list=fakeNode(),view=Object.create(ObjectsView.prototype);
  view.find=selector=>selector==='.objects-list'?list:{value:''};
  view.items=[{id:'so101',name:'SO-101 robot',category:'Robot',tasks:[],variants:[{name:'SO-101 robot',label:'Original materials'}],thumbnail_url:'/api/objects/so101/thumbnail?revision=1'}];
  view.api=()=>assert.fail('Cards must not download full geometry');view.renderList();
  const image=list.children[0].children[0];
  assert.equal(image.tag,'img');assert.equal(image.src,view.items[0].thumbnail_url);assert.equal(image.loading,'lazy');
 }finally{globalThis.document=previous;}
});
