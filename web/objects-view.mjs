/** Current MuJoCo visual geometry, rendered locally without external libraries. */
export const materialColor=rgba=>'rgb('+rgba.slice(0,3).map(v=>Math.round(Math.max(0,Math.min(1,v))*255)).join(',')+')';
export const paletteBackground=materials=>materials.length===1?materialColor(materials[0]):`linear-gradient(135deg,${materials.map((rgba,i)=>`${materialColor(rgba)} ${i/materials.length*100}% ${(i+1)/materials.length*100}%`).join(',')})`;
export const formatDimensions=values=>values.map(value=>Number((value*1000).toFixed(1))).join(' × ')+' mm';
export function filterObjects(items,query){
 const needle=query.trim().toLowerCase();
 return items.filter(item=>[item.name,item.category,...(item.variants||[]).map(v=>v.name+' '+v.label),...item.tasks.map(task=>task.instruction)].join(' ').toLowerCase().includes(needle));
}
export function triangleBuffer(part,scale){
 const out=new Float32Array(part.faces.length*6),v=part.vertices;
 for(let f=0;f<part.faces.length;f+=3){
  const a=part.faces[f]*3,b=part.faces[f+1]*3,c=part.faces[f+2]*3;
  const ux=v[b]-v[a],uy=v[b+1]-v[a+1],uz=v[b+2]-v[a+2],vx=v[c]-v[a],vy=v[c+1]-v[a+1],vz=v[c+2]-v[a+2];
  let nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;const length=Math.hypot(nx,ny,nz)||1;nx/=length;ny/=length;nz/=length;
  for(let j=0;j<3;j++){const index=part.faces[f+j]*3,offset=(f+j)*6;out.set([v[index]/scale,v[index+1]/scale,v[index+2]/scale,nx,ny,nz],offset);}
 }
 return out;
}
class ObjectPreview{
 constructor(canvas){
  this.canvas=canvas;this.parts=[];this.shaded=false;this.yaw=-.9;this.pitch=.45;this.zoom=1.4;
  const gl=canvas.getContext('webgl',{antialias:true,alpha:true});if(!gl)throw new Error('3D preview requires WebGL. Model dimensions and task usage remain available.');this.gl=gl;
  const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error('3D preview could not initialize.');return s;};
  const vs=shader(gl.VERTEX_SHADER,'attribute vec3 position;attribute vec3 normal;uniform vec3 right;uniform vec3 up;uniform vec3 eye;uniform vec2 scale;varying vec3 n;void main(){gl_Position=vec4(dot(position,right)*scale.x,dot(position,up)*scale.y,-dot(position,eye)/3.0,1.0);n=normal;}');
  const fs=shader(gl.FRAGMENT_SHADER,'precision mediump float;uniform vec4 color;uniform float shaded;varying vec3 n;void main(){float light=0.48+0.52*abs(dot(normalize(n),normalize(vec3(-0.4,-0.6,1.0))));gl_FragColor=vec4(color.rgb*mix(1.0,light,shaded),1.0);}');
  this.program=gl.createProgram();gl.attachShader(this.program,vs);gl.attachShader(this.program,fs);gl.linkProgram(this.program);gl.deleteShader(vs);gl.deleteShader(fs);
  if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw new Error('3D preview could not initialize.');
  this.uniforms=Object.fromEntries(['right','up','eye','scale','color','shaded'].map(name=>[name,gl.getUniformLocation(this.program,name)]));
  this.position=gl.getAttribLocation(this.program,'position');this.normal=gl.getAttribLocation(this.program,'normal');
  canvas.addEventListener('pointerdown',event=>{this.pointer={id:event.pointerId,x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);});
  canvas.addEventListener('pointermove',event=>{if(!this.pointer)return;this.yaw-=(event.clientX-this.pointer.x)*.009;this.pitch=Math.max(-1.45,Math.min(1.45,this.pitch+(event.clientY-this.pointer.y)*.009));this.pointer={id:event.pointerId,x:event.clientX,y:event.clientY};this.draw();});
  for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>{this.pointer=null;});
  canvas.addEventListener('wheel',event=>{event.preventDefault();this.zoom=Math.max(.5,Math.min(5,this.zoom*Math.exp(-event.deltaY*.001)));this.draw();},{passive:false});
  canvas.addEventListener('keydown',event=>{const actions={ArrowLeft:()=>this.yaw-=.15,ArrowRight:()=>this.yaw+=.15,ArrowUp:()=>this.pitch=Math.min(1.45,this.pitch+.15),ArrowDown:()=>this.pitch=Math.max(-1.45,this.pitch-.15),'+':()=>this.zoom=Math.min(5,this.zoom*1.15),'-':()=>this.zoom=Math.max(.5,this.zoom/1.15),Home:()=>this.reset()};if(actions[event.key]){event.preventDefault();actions[event.key]();this.draw();}});
  this.observer=new ResizeObserver(()=>this.draw());this.observer.observe(canvas);
 }
 clear(){for(const part of this.parts)this.gl.deleteBuffer(part.buffer);this.parts=[];this.draw();}
 setMesh(mesh){
  this.clear();const gl=this.gl,scale=Math.max(...mesh.dimensions_m)||1;
  this.parts=mesh.parts.map(part=>{const values=triangleBuffer(part,scale),buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,values,gl.STATIC_DRAW);return {buffer,count:values.length/6,rgba:part.rgba};});this.reset();
 }
 reset(){this.yaw=-.9;this.pitch=.45;this.zoom=1.4;this.draw();}
 draw(){
  const gl=this.gl,canvas=this.canvas,r=canvas.getBoundingClientRect();if(!r.width||!r.height)return;
  const dpr=Math.min(globalThis.devicePixelRatio||1,2),width=Math.round(r.width*dpr),height=Math.round(r.height*dpr);
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}gl.viewport(0,0,width,height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.useProgram(this.program);
  const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
  gl.uniform3fv(this.uniforms.right,[-sy,cy,0]);gl.uniform3fv(this.uniforms.up,[-sp*cy,-sp*sy,cp]);gl.uniform3fv(this.uniforms.eye,[cp*cy,cp*sy,sp]);gl.uniform2fv(this.uniforms.scale,[this.zoom/Math.max(1,width/height),this.zoom/Math.max(1,height/width)]);
  gl.uniform1f(this.uniforms.shaded,this.shaded?1:0);
  for(const part of this.parts){gl.bindBuffer(gl.ARRAY_BUFFER,part.buffer);gl.enableVertexAttribArray(this.position);gl.enableVertexAttribArray(this.normal);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,24,0);gl.vertexAttribPointer(this.normal,3,gl.FLOAT,false,24,12);gl.uniform4fv(this.uniforms.color,part.rgba);gl.drawArrays(gl.TRIANGLES,0,part.count);}
 }
 setLighting(value){this.shaded=value;this.draw();}
 destroy(){this.observer.disconnect();this.clear();this.gl.deleteProgram(this.program);}
}
const element=(tag,className,text)=>{const node=document.createElement(tag);if(className)node.className=className;if(text!==undefined)node.textContent=text;return node;};
export class ObjectsView{
 constructor(host,api){
  this.host=host;this.api=api;this.active=false;this.version=0;this.items=[];this.meshes=new Map();this.preview=null;this.lighting='material';
  host.classList.add('objects-view');
  host.innerHTML=`<div class="objects-heading"><div><div class="objects-eyebrow">MODEL LIBRARY</div><h2>Model library</h2><p>Explore the models in your simulation and the tasks that use them.</p></div><div class="objects-summary" aria-live="polite"></div></div><div class="objects-workspace"><aside class="objects-library"><label for="objects-search">Find a model or task</label><input id="objects-search" type="search" placeholder="Search objects…" autocomplete="off"><div class="objects-list"></div></aside><section class="objects-detail"><div class="objects-detail-heading"><div><span class="objects-category"></span><h2 class="objects-name">Select an object</h2></div><button type="button" class="objects-reset">Reset view</button></div><div class="objects-controls"><fieldset class="objects-color-control"><legend>Color variant</legend><div id="objects-variant" class="objects-swatch-buttons" role="group" aria-label="Color variant"></div><span class="objects-color-label"></span></fieldset><fieldset class="objects-appearance-control"><legend>Preview appearance</legend><div id="objects-lighting" class="objects-appearance-buttons" role="group" aria-label="Preview appearance"></div></fieldset><div class="objects-material-palette" aria-label="Materials in this model"></div><div class="objects-sources"></div><span class="objects-material-note"></span></div><div class="objects-stage"><canvas tabindex="0" aria-label="Interactive 3D model. Drag or use arrow keys to orbit; scroll or use plus and minus to zoom; Home to reset."></canvas><div class="objects-preview-status" role="status"></div><div class="objects-preview-help">Drag to orbit · Scroll to zoom</div><span class="objects-preview-label">ACTUAL MODEL GEOMETRY</span></div><div class="objects-facts"></div><p class="objects-description"></p><div class="objects-usage"><h3>Used in tasks</h3><div class="objects-tasks"></div></div></section></div><p class="objects-note" role="status"></p>`;
  this.find=selector=>host.querySelector(selector);
  this.find('#objects-search').addEventListener('input',()=>this.renderList());
  this.find('.objects-reset').onclick=()=>this.preview?.reset();

 }
 async load(){
  this.active=true;const version=++this.version;this.find('.objects-note').textContent='Loading the current simulation models…';
  try{
   const data=await this.api('/api/objects');if(!this.active||version!==this.version)return;
   if(this.revision!==data.revision){this.meshes.clear();this.revision=data.revision;}
   this.items=data.objects;this.find('.objects-summary').textContent=`${data.objects.length} models · ${data.source_count||3} sources`;this.find('.objects-note').textContent=data.note;
   this.renderList();const item=this.items.find(item=>item.id===this.selected)||this.items[0];if(item)await this.select(item.id);
  }catch(error){if(this.active&&version===this.version){this.find('.objects-note').textContent=error.message;const retry=element('button','','Retry loading models');retry.type='button';retry.onclick=()=>this.load();this.find('.objects-note').append(' ',retry);}}
 }
 renderList(){
  const list=this.find('.objects-list');list.replaceChildren();const items=filterObjects(this.items,this.find('#objects-search').value);
  if(!items.length){list.append(element('p','objects-empty','No models match your search.'));return;}
  for(const item of items){const button=element('button','objects-item');button.type='button';button.dataset.objectId=item.id;button.setAttribute('aria-pressed',String(item.id===this.selected));
   const mark=element('img','objects-mark');mark.src=item.thumbnail_url;mark.alt='';mark.width=56;mark.height=56;mark.loading='lazy';mark.decoding='async';
   const text=element('span','objects-item-text');text.append(element('strong','',item.name),element('span','',item.category),element('small','',`${item.variants.length} color variant${item.variants.length===1?'':'s'}`));button.append(mark,text);button.onclick=()=>this.select(item.id);list.append(button);}
 }
 async select(identifier){
  const item=this.items.find(item=>item.id===identifier);if(!item)return;this.selected=identifier;this.renderList();
  const previous=this.variant;
  await this.selectVariant(item.variants.some(v=>v.id===previous)?previous:item.variants[0].id);
 }
 renderControls(model,item){
  const variants=this.find('#objects-variant');variants.replaceChildren();
  for(const variant of model.variants){
   const button=element('button','objects-color-button');button.type='button';button.dataset.variantId=variant.id;
   button.style.background=paletteBackground(variant.materials||[variant.rgba]);
   button.setAttribute('aria-label',variant.label);button.setAttribute('aria-pressed',String(variant.id===item.id));
   button.title=variant.label+' · '+(variant.sources||[]).map(s=>s.label).join(', ');
   button.onclick=()=>this.selectVariant(variant.id);variants.append(button);
  }
  this.find('.objects-color-label').textContent=item.label;
  const appearance=this.find('#objects-lighting');appearance.replaceChildren();
  for(const [value,label] of [['material','Base color'],['studio','Lit preview']]){
   const choice=element('div','objects-appearance-choice'),button=element('button','objects-appearance-button');
   button.type='button';button.dataset.appearance=value;button.setAttribute('aria-label',label);button.title=label;
   button.setAttribute('aria-pressed',String(this.lighting===value));
   const color=element('span','objects-appearance-color');color.style.background=paletteBackground(item.materials||[item.rgba]);button.append(color);
   button.onclick=()=>{this.lighting=value;this.preview?.setLighting(value==='studio');this.renderControls(model,item);this.appearanceNote();};
   choice.append(button,element('span','',label));appearance.append(choice);
  }
  const palette=this.find('.objects-material-palette');palette.replaceChildren();
  if(item.materials?.length>1){
   palette.append(element('span','','Original part colors'));
   for(const rgba of item.materials){const swatch=element('span','objects-part-swatch');swatch.style.background=materialColor(rgba);swatch.title=`Part material RGBA: ${rgba.map(v=>Number(v.toFixed(3))).join(', ')}`;swatch.setAttribute('aria-label',swatch.title);palette.append(swatch);}
  }
  const sources=this.find('.objects-sources');sources.replaceChildren();sources.append(element('strong','','Used by '));
  for(const source of item.sources||[]){const badge=element('span','objects-source',source.label);badge.title=source.path;sources.append(badge);}
 }
 appearanceNote(){this.find('.objects-material-note').textContent=this.lighting==='studio'?'Preview lighting only. Dataset lighting and camera color processing can differ.':'Unlit simulation base colors. Dataset images also include lighting and may include camera color matching.';}
 async selectVariant(identifier){
  const model=this.items.find(item=>item.id===this.selected),item=model?.variants.find(v=>v.id===identifier);if(!item)return;this.variant=identifier;const version=++this.version;
  this.renderControls(model,item);this.appearanceNote();
  this.find('.objects-name').textContent=model.name+(item.label==='Default'?'':' · '+item.label);this.find('.objects-category').textContent=item.category;this.find('.objects-description').textContent=item.description;
  const facts=this.find('.objects-facts');facts.replaceChildren();for(const [label,value] of [[item.dimension_note,formatDimensions(item.dimensions_m)],['Visual geometry',`${item.triangle_count.toLocaleString()} triangles · ${item.part_count} parts`]]){const fact=element('div');fact.append(element('span','',label),element('strong','',value));facts.append(fact);}
  const tasks=this.find('.objects-tasks');tasks.replaceChildren();for(const task of item.tasks){const row=element('div','objects-task');row.append(element('span','',task.instruction),element('span','objects-role',task.role));tasks.append(row);}
  const status=this.find('.objects-preview-status');status.textContent='Loading model geometry…';this.preview?.clear();
  try{
   if(!this.preview)this.preview=new ObjectPreview(this.find('canvas'));
   let mesh=this.meshes.get(identifier);if(!mesh){mesh=await this.api(item.mesh_url);if(!this.active||version!==this.version)return;this.meshes.set(identifier,mesh);}
   if(!this.active||version!==this.version)return;this.preview.setMesh(mesh);this.preview.setLighting(this.lighting==='studio');this.find('canvas').setAttribute('aria-label',`${item.name} 3D preview. Drag or arrow keys to orbit; scroll or plus and minus to zoom; Home to reset.`);status.textContent='';
  }catch(error){if(this.active&&version===this.version){status.textContent=error.message;const retry=element('button','','Retry preview');retry.type='button';retry.onclick=()=>this.selectVariant(identifier);status.append(retry);}}
 }
 suspend(){this.active=false;this.version++;}
 destroy(){this.suspend();this.preview?.destroy();this.meshes.clear();}
}
