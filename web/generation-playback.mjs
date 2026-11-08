// Both camera canvases and every plot consume a single decoded media frame.
export function frameAtTime(time,fps,frames){return Math.max(0,Math.min(frames-1,Math.floor(time*fps+1e-5)));}
export class EpisodePlayback{
 constructor(video,{onFrame=()=>{},onState=()=>{},onError=()=>{}}={}){
  this.video=video;this.onFrame=onFrame;this.onState=onState;this.onError=onError;this.episode=null;this.ready=false;this.callback=null;this.listeners=[];this.version=0;
  for(const name of ['loadeddata','canplay','play','pause','seeked','seeking','timeupdate','ended','waiting','playing','error']){
   const fn=()=>{
    this.ready=!!this.episode&&video.readyState>=2&&!video.seeking;
    if(name==='error'){this.ready=false;this.cancel();if(this.episode)this.onError('The synchronized video could not be decoded. Select the episode again to retry.');}
    if(this.ready&&(video.paused||!video.requestVideoFrameCallback||['loadeddata','seeked','ended'].includes(name)))this.paint(video.currentTime);
    this.emit();if(video.paused||!this.ready)this.cancel();else this.schedule();
   };
   video.addEventListener(name,fn);this.listeners.push([name,fn]);
  }
 }
 emit(){this.onState({ready:this.ready,paused:this.video.paused,time:this.video.currentTime,ended:this.video.ended});}
 paint(time){if(this.ready&&this.episode)this.onFrame(frameAtTime(time,this.episode.fps,this.episode.frames),time);}
 cancel(){if(this.callback!==null){if(this.video.cancelVideoFrameCallback)this.video.cancelVideoFrameCallback(this.callback);else globalThis.cancelAnimationFrame?.(this.callback);this.callback=null;}}
 schedule(){
  if(this.callback!==null||!this.ready||this.video.paused)return;
  const version=this.version;
  const tick=(_,metadata)=>{this.callback=null;if(version!==this.version)return;if(!this.video.seeking)this.paint(metadata?.mediaTime??this.video.currentTime);this.schedule();};
  this.callback=this.video.requestVideoFrameCallback?this.video.requestVideoFrameCallback(tick):globalThis.requestAnimationFrame?.(tick)??null;
 }
 setEpisode(episode){
  if(this.episode?.url===episode?.url&&this.episode?.frames===episode?.frames&&this.episode?.fps===episode?.fps)return;
  this.version++;this.cancel();this.episode=null;this.ready=false;this.video.pause();
  this.episode=episode;
  if(episode)this.video.src=episode.url;else this.video.removeAttribute('src');
  this.video.load();this.emit();
 }
 pause(){this.cancel();this.video.pause();}
 seekFrame(index){
  if(!this.episode||this.video.readyState<1)return;
  this.pause();const frame=Math.max(0,Math.min(this.episode.frames-1,Math.round(index)));
  this.video.currentTime=(frame+.001)/this.episode.fps;
 }
 setRate(rate){if([.25,.5,1,1.5,2].includes(Number(rate)))this.video.playbackRate=Number(rate);}
 async toggle(){
  if(!this.ready||!this.episode)return false;
  if(!this.video.paused){this.pause();return true;}
  if(this.video.ended||frameAtTime(this.video.currentTime,this.episode.fps,this.episode.frames)===this.episode.frames-1)this.seekFrame(0);
  const version=this.version;
  try{await this.video.play();if(version===this.version)this.schedule();return version===this.version;}
  catch(error){if(version===this.version){this.pause();this.onError(error.message);}return false;}
 }
 destroy(){this.pause();this.version++;for(const [name,fn]of this.listeners)this.video.removeEventListener(name,fn);}
}
