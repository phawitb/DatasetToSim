// One media clock repaints both viewports. A status refresh must never reload it.
export class Playback {
  constructor(video,{onFrame=()=>{},onState=()=>{},onError=()=>{}}={}) {
    this.video=video;this.onFrame=onFrame;this.onState=onState;this.onError=onError;
    this.source=null;this.ready=false;this.frameRequest=null;this.listeners=[];
    for(const event of ['loadeddata','canplay','play','pause','timeupdate','seeked','ended','error']) {
      const handler=()=>{
        this.ready=!!this.source&&video.readyState>=2;
        if(event==='error'){this.ready=false;if(this.source)this.onError('Video could not be decoded. Please retry.');}
        if(this.ready){try{this.onFrame();}catch(error){this.onError(error.message);}}
        this.onState({ready:this.ready,paused:video.paused,time:video.currentTime,ended:video.ended});
        this.schedule();
      };
      video.addEventListener(event,handler);this.listeners.push([event,handler]);
    }
  }
  cancel(){
    if(this.frameRequest!==null){
      if(this.video.cancelVideoFrameCallback)this.video.cancelVideoFrameCallback(this.frameRequest);
      else if(globalThis.cancelAnimationFrame)cancelAnimationFrame(this.frameRequest);
      this.frameRequest=null;
    }
  }
  schedule(){
    if(this.video.paused||!this.ready||this.frameRequest!==null)return;
    const tick=()=>{this.frameRequest=null;try{this.onFrame();}catch(e){this.onError(e.message);}finally{this.schedule();}};
    if(this.video.requestVideoFrameCallback)this.frameRequest=this.video.requestVideoFrameCallback(tick);
    else if(globalThis.requestAnimationFrame)this.frameRequest=requestAnimationFrame(tick);
  }
  setSource(url){
    url=url||null;if(this.source===url)return;
    this.cancel();this.video.pause();this.source=url;this.ready=false;
    if(url)this.video.src=url;else this.video.removeAttribute('src');
    this.video.load();this.onState({ready:false,paused:true,time:0,ended:false});
  }
  async toggle(){
    if(!this.ready||!this.source)return false;
    if(!this.video.paused){this.video.pause();return true;}
    if(this.video.ended||this.video.currentTime>=this.video.duration-.02)this.video.currentTime=0;
    try{await this.video.play();this.schedule();return true;}catch(e){this.onError(e.message);return false;}
  }
  destroy(){this.cancel();for(const [e,f]of this.listeners)this.video.removeEventListener(e,f);}
}
