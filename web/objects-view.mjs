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
const GRASP_RGBA=[.98,.45,.12,1],PAD_RGBA=[.92,.22,.12,1],SURFACE_RGBA=[.15,.62,.85,1];
function box(centre,half,yaw,rgba){
 const c=Math.cos(yaw),s=Math.sin(yaw),corners=[];
 for(const z of [-1,1])for(const [x,y] of [[-1,-1],[1,-1],[1,1],[-1,1]]){const lx=x*half[0],ly=y*half[1];corners.push(centre[0]+c*lx-s*ly,centre[1]+s*lx+c*ly,centre[2]+z*half[2]);}
 return {vertices:corners,faces:[0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7],rgba,overlay:true};
}
function ring(centre,inner,outer,rgba,n=48){
 const vertices=[],faces=[];
 for(let i=0;i<n;i++){const a=i/n*2*Math.PI;for(const r of [inner,outer])vertices.push(centre[0]+r*Math.cos(a),centre[1]+r*Math.sin(a),centre[2]);}
 for(let i=0;i<n;i++){const a=2*i,b=2*((i+1)%n);faces.push(a,b,a+1,b,b+1,a+1);}
 return {vertices,faces,rgba,overlay:true};
}
/** Marker geometry for one registry entry, in the model's own frame (metres). */
export function contactParts(entry){
 const parts=[],pad=.003;
 for(const grasp of entry?.grasps||[]){
  const c=grasp.centre_m,half=grasp.width_m/2+pad/2;
  if(grasp.pads==='fixed_in_hole'||grasp.jaw_axis_deg==='radial'){
   const radius=Math.hypot(c[0],c[1]);
   parts.push(ring([0,0,c[2]],Math.max(0,radius-.0015),radius+.0015,GRASP_RGBA));
   parts.push(box([radius-half,0,c[2]],[pad/2,.004,.005],0,PAD_RGBA),box([radius+half,0,c[2]],[pad/2,.004,.005],0,PAD_RGBA));
  }else for(const deg of grasp.jaw_axis_deg){
   const a=deg*Math.PI/180,dx=Math.cos(a)*half,dy=Math.sin(a)*half;
   parts.push(box([c[0]+dx,c[1]+dy,c[2]],[pad/2,.004,.005],a,PAD_RGBA),box([c[0]-dx,c[1]-dy,c[2]],[pad/2,.004,.005],a,PAD_RGBA));
   parts.push(box(c,[half,.0007,.0007],a,GRASP_RGBA));
  }
 }
 for(const surface of entry?.place_surfaces||[]){
  const c=[surface.centre_m[0],surface.centre_m[1],surface.centre_m[2]+.0006];
  if(surface.half_extents_m)parts.push(box(c,[surface.half_extents_m[0],surface.half_extents_m[1],.0003],0,SURFACE_RGBA));
  else parts.push(ring(c,surface.inner_radius_m||0,surface.radius_m,SURFACE_RGBA));
 }
 return parts;
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
 setMesh(mesh,overlays=[]){
  this.clear();const gl=this.gl,scale=Math.max(...mesh.dimensions_m)||1;this.scale=scale;
  this.parts=[...mesh.parts,...overlays].map(part=>{const values=triangleBuffer(part,scale),buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,values,gl.STATIC_DRAW);return {buffer,count:values.length/6,rgba:part.rgba,overlay:!!part.overlay};});this.reset();
 }
 reset(){this.yaw=-.9;this.pitch=.45;this.zoom=1.4;this.draw();}
 draw(){
  const gl=this.gl,canvas=this.canvas,r=canvas.getBoundingClientRect();if(!r.width||!r.height)return;
  const dpr=Math.min(globalThis.devicePixelRatio||1,2),width=Math.round(r.width*dpr),height=Math.round(r.height*dpr);
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}gl.viewport(0,0,width,height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.useProgram(this.program);
  const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
  gl.uniform3fv(this.uniforms.right,[-sy,cy,0]);gl.uniform3fv(this.uniforms.up,[-sp*cy,-sp*sy,cp]);gl.uniform3fv(this.uniforms.eye,[cp*cy,cp*sy,sp]);gl.uniform2fv(this.uniforms.scale,[this.zoom/Math.max(1,width/height),this.zoom/Math.max(1,height/width)]);
  gl.uniform1f(this.uniforms.shaded,this.shaded?1:0);
  // Contact markers draw last and on top, so areas inside the model stay visible.
  this.parts.sort((a,b)=>a.overlay-b.overlay);
  for(const part of this.parts){if(part.overlay)gl.disable(gl.DEPTH_TEST);gl.bindBuffer(gl.ARRAY_BUFFER,part.buffer);gl.enableVertexAttribArray(this.position);gl.enableVertexAttribArray(this.normal);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,24,0);gl.vertexAttribPointer(this.normal,3,gl.FLOAT,false,24,12);gl.uniform4fv(this.uniforms.color,part.rgba);gl.drawArrays(gl.TRIANGLES,0,part.count);}
 }
 setOverlays(overlays){
  const gl=this.gl;for(const part of this.parts.filter(p=>p.overlay))gl.deleteBuffer(part.buffer);
  this.parts=this.parts.filter(p=>!p.overlay).concat(overlays.map(part=>{const values=triangleBuffer(part,this.scale||1),buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,values,gl.STATIC_DRAW);return {buffer,count:values.length/6,rgba:part.rgba,overlay:true};}));this.draw();
 }
 setLighting(value){this.shaded=value;this.draw();}
 destroy(){this.observer.disconnect();this.clear();this.gl.deleteProgram(this.program);}
}
const element=(tag,className,text)=>{const node=document.createElement(tag);if(className)node.className=className;if(text!==undefined)node.textContent=text;return node;};
export class ObjectsView{
 constructor(host,api){
  this.host=host;this.api=api;this.active=false;this.version=0;this.items=[];this.meshes=new Map();this.preview=null;this.lighting='material';
  host.classList.add('objects-view');
  host.innerHTML=`<div class="objects-heading"><div><div class="objects-eyebrow">MODEL LIBRARY</div><h2>Model library</h2><p>Explore the models in your simulation and the tasks that use them.</p></div><div class="objects-summary" aria-live="polite"></div></div><div class="objects-workspace"><aside class="objects-library"><label for="objects-search">Find a model or task</label><input id="objects-search" type="search" placeholder="Search objects…" autocomplete="off"><div class="objects-list"></div></aside><section class="objects-detail"><div class="objects-detail-heading"><div><span class="objects-category"></span><h2 class="objects-name">Select an object</h2></div><button type="button" class="objects-reset">Reset view</button></div><div class="objects-controls"><fieldset class="objects-color-control"><legend>Color variant</legend><div id="objects-variant" class="objects-swatch-buttons" role="group" aria-label="Color variant"></div><span class="objects-color-label"></span></fieldset><fieldset class="objects-appearance-control"><legend>Preview appearance</legend><div id="objects-lighting" class="objects-appearance-buttons" role="group" aria-label="Preview appearance"></div></fieldset><div class="objects-material-palette" aria-label="Materials in this model"></div><div class="objects-sources"></div><span class="objects-material-note"></span></div><div class="objects-stage"><canvas tabindex="0" aria-label="Interactive 3D model. Drag or use arrow keys to orbit; scroll or use plus and minus to zoom; Home to reset."></canvas><div class="objects-preview-status" role="status"></div><div class="objects-preview-help">Drag to orbit · Scroll to zoom</div><span class="objects-preview-label">ACTUAL MODEL GEOMETRY</span></div><div class="objects-facts"></div><div class="objects-contacts hidden"><div class="objects-contacts-heading"><h3>Contact areas</h3><span class="objects-contacts-legend"><i class="grasp"></i>Grasp · jaw pads <i class="surface"></i>Place surface</span></div><p class="objects-contacts-note">Where the jaws close and where other objects can be set down, in this model's own frame. New Run plans every pick and place from these values.</p><div class="objects-contacts-form"></div><div class="objects-contacts-actions"><button type="button" class="objects-contacts-save button">Save contact areas</button><button type="button" class="objects-contacts-reset text-button">Discard changes</button><span class="objects-contacts-status" role="status"></span></div></div><p class="objects-description"></p><div class="objects-usage"><h3>Used in tasks</h3><div class="objects-tasks"></div></div></section></div><p class="objects-note" role="status"></p>`;
  this.find=selector=>host.querySelector(selector);
  this.find('#objects-search').addEventListener('input',()=>this.renderList());
  this.find('.objects-reset').onclick=()=>this.preview?.reset();
  this.find('.objects-contacts-save').onclick=()=>this.saveContacts();
  this.find('.objects-contacts-reset').onclick=()=>{this.draft=null;this.renderContacts();};

 }
 async load(){
  this.active=true;const version=++this.version;this.find('.objects-note').textContent='Loading the current simulation models…';
  try{
   const [data,registry]=await Promise.all([this.api('/api/objects'),this.api('/api/affordances').catch(()=>null)]);if(!this.active||version!==this.version)return;this.registry=registry;
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
 contactKind(){
  const ids=[this.selected,this.variant];
  return Object.entries(this.registry?.objects||{}).find(([kind,entry])=>ids.includes(kind)||ids.some(id=>(entry.library_ids||[]).includes(id)))?.[0];
 }
 contactEntry(){const kind=this.contactKind();return kind?(this.draft?.kind===kind?this.draft.entry:this.registry.objects[kind]):null;}
 renderContacts(){
  const panel=this.find('.objects-contacts'),kind=this.contactKind();panel.classList.toggle('hidden',!kind);this.preview?.setOverlays(kind?contactParts(this.contactEntry()):[]);if(!kind)return;
  const entry=this.contactEntry(),form=this.find('.objects-contacts-form');form.replaceChildren();
  const mm=v=>Number((v*1000).toFixed(2)),edit=(update)=>{if(this.draft?.kind!==kind)this.draft={kind,entry:structuredClone(this.registry.objects[kind])};update(this.draft.entry);this.preview?.setOverlays(contactParts(this.draft.entry));this.find('.objects-contacts-status').textContent='Unsaved changes';};
  const field=(label,value,onchange,type='number')=>{const wrap=element('label','objects-contact-field');wrap.append(element('span','',label));const input=element('input');input.type=type;if(type==='number')input.step='0.5';input.value=value;input.onchange=()=>onchange(input.value);wrap.append(input);return wrap;};
  const select=(label,value,options,onchange)=>{const wrap=element('label','objects-contact-field');wrap.append(element('span','',label));const input=element('select');for(const o of options)input.append(new Option(o,o));input.value=value;input.onchange=()=>onchange(input.value);wrap.append(input);return wrap;};
  entry.grasps.forEach((grasp,i)=>{const box=element('fieldset','objects-contact');box.append(element('legend','','Grasp · '+grasp.name));
   const at=(k,v)=>edit(e=>{e.grasps[i][k]=v;});
   ['x','y','z'].forEach((axis,j)=>box.append(field(`Centre ${axis} (mm)`,mm(grasp.centre_m[j]),v=>edit(e=>{e.grasps[i].centre_m[j]=Number(v)/1000;}))));
   box.append(field('Pad height (mm)',mm(grasp.pad_height_m),v=>at('pad_height_m',Number(v)/1000)),field('Width (mm)',mm(grasp.width_m),v=>at('width_m',Number(v)/1000)),
    field('Jaw axis (° or radial)',grasp.jaw_axis_deg==='radial'?'radial':grasp.jaw_axis_deg.join(', '),v=>at('jaw_axis_deg',v.trim()==='radial'?'radial':v.split(',').map(Number)),'text'),
    select('Pads',grasp.pads,['between','fixed_in_hole'],v=>at('pads',v)));
   if(grasp.note)box.append(element('p','objects-contact-note',grasp.note));form.append(box);});
  (entry.place_surfaces||[]).forEach((surface,i)=>{const box=element('fieldset','objects-contact surface');box.append(element('legend','','Place surface · '+surface.name));
   const at=(k,v)=>edit(e=>{e.place_surfaces[i][k]=v;});
   ['x','y','z'].forEach((axis,j)=>box.append(field(`Centre ${axis} (mm)`,mm(surface.centre_m[j]),v=>edit(e=>{e.place_surfaces[i].centre_m[j]=Number(v)/1000;}))));
   if(surface.half_extents_m)['x','y'].forEach((axis,j)=>box.append(field(`Half size ${axis} (mm)`,mm(surface.half_extents_m[j]),v=>edit(e=>{e.place_surfaces[i].half_extents_m[j]=Number(v)/1000;}))));
   else{box.append(field('Radius (mm)',mm(surface.radius_m),v=>at('radius_m',Number(v)/1000)));if('inner_radius_m' in surface)box.append(field('Hole radius (mm)',mm(surface.inner_radius_m),v=>at('inner_radius_m',Number(v)/1000)));}
   box.append(select('Type',surface.type,['top','inside'],v=>at('type',v)),select('Release',surface.release,['lower','above_rim'],v=>at('release',v)));form.append(box);});
  if(this.draft?.kind!==kind)this.find('.objects-contacts-status').textContent='';
 }
 async saveContacts(){
  const kind=this.contactKind();if(!kind||this.draft?.kind!==kind)return;const status=this.find('.objects-contacts-status');
  try{status.textContent='Saving…';const registry=structuredClone(this.registry);registry.objects[kind]=this.draft.entry;this.registry=await this.api('/api/affordances',registry);this.draft=null;status.textContent='Saved · New Run uses these contact areas from the next plan';this.renderContacts();}
  catch(error){status.textContent=error.message;}
 }
 async selectVariant(identifier){
  const model=this.items.find(item=>item.id===this.selected),item=model?.variants.find(v=>v.id===identifier);if(!item)return;this.variant=identifier;const version=++this.version;
  this.renderControls(model,item);this.appearanceNote();
  this.find('.objects-name').textContent=model.name+(item.label==='Default'?'':' · '+item.label);this.find('.objects-category').textContent=item.category;this.find('.objects-description').textContent=item.description;
  const facts=this.find('.objects-facts');facts.replaceChildren();for(const [label,value] of [[item.dimension_note,formatDimensions(item.dimensions_m)],['Visual geometry',`${item.triangle_count.toLocaleString()} triangles · ${item.part_count} parts`]]){const fact=element('div');fact.append(element('span','',label),element('strong','',value));facts.append(fact);}
  const tasks=this.find('.objects-tasks');tasks.replaceChildren();for(const task of item.tasks){const row=element('div','objects-task');row.append(element('span','',task.instruction),element('span','objects-role',task.role));tasks.append(row);}
  if(!item.tasks.length)tasks.append(element('p','objects-empty','Not used in any task yet.'));
  const status=this.find('.objects-preview-status');status.textContent='Loading model geometry…';this.preview?.clear();
  try{
   if(!this.preview)this.preview=new ObjectPreview(this.find('canvas'));
   let mesh=this.meshes.get(identifier);if(!mesh){mesh=await this.api(item.mesh_url);if(!this.active||version!==this.version)return;this.meshes.set(identifier,mesh);}
   if(!this.active||version!==this.version)return;this.preview.setMesh(mesh,this.contactKind()?contactParts(this.contactEntry()):[]);this.renderContacts();this.preview.setLighting(this.lighting==='studio');this.find('canvas').setAttribute('aria-label',`${item.name} 3D preview. Drag or arrow keys to orbit; scroll or plus and minus to zoom; Home to reset.`);status.textContent='';
  }catch(error){if(this.active&&version===this.version){status.textContent=error.message;const retry=element('button','','Retry preview');retry.type='button';retry.onclick=()=>this.selectVariant(identifier);status.append(retry);}}
 }
 suspend(){this.active=false;this.version++;}
 destroy(){this.suspend();this.preview?.destroy();this.meshes.clear();}
}
