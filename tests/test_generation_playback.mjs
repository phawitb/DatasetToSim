import test from 'node:test';
import assert from 'node:assert/strict';
import {EpisodePlayback,frameAtTime} from '../web/generation-playback.mjs';
class Video extends EventTarget{
 constructor(){super();this.paused=true;this.readyState=0;this.currentTime=0;this.duration=2;this.ended=false;this.seeking=false;this.loads=0;this.callbacks=new Map();this.id=0;}
 load(){this.loads++;this.readyState=0;this.currentTime=0;}
 removeAttribute(){this.src='';}
 pause(){this.paused=true;this.dispatchEvent(new Event('pause'));}
 play(){this.paused=false;this.dispatchEvent(new Event('play'));return Promise.resolve();}
 ready(){this.readyState=4;this.dispatchEvent(new Event('loadeddata'));}
 requestVideoFrameCallback(fn){this.callbacks.set(++this.id,fn);return this.id;}
 cancelVideoFrameCallback(id){this.callbacks.delete(id);}
 frame(t){const callbacks=[...this.callbacks.values()];this.callbacks.clear();for(const fn of callbacks)fn(0,{mediaTime:t});}
}
const descriptor={url:'paired.mp4',frames:60,fps:30};
test('sample index is causal, exact at frame boundaries and clamped at end',()=>{
 assert.equal(frameAtTime(0,30,60),0);assert.equal(frameAtTime(1/30,30,60),1);
 assert.equal(frameAtTime(1.999,30,60),59);assert.equal(frameAtTime(2,30,60),59);
});
test('one decoded media timestamp drives both images and state/action sample',async()=>{
 const video=new Video(),seen=[];const player=new EpisodePlayback(video,{onFrame:(frame,time)=>seen.push([frame,time])});
 player.setEpisode(descriptor);video.ready();await player.toggle();video.currentTime=.8;video.frame(.5);
 assert.deepEqual(seen.at(-1),[15,.5]);player.destroy();
});
test('polling same episode never reloads or pauses playing media',async()=>{
 const video=new Video(),player=new EpisodePlayback(video);player.setEpisode(descriptor);video.ready();await player.toggle();
 video.currentTime=.7;const loads=video.loads;player.setEpisode({...descriptor});
 assert.equal(video.loads,loads);assert.equal(video.currentTime,.7);assert.equal(video.paused,false);player.destroy();
});
test('seek updates once decoded, step and replay clamp to valid rows',async()=>{
 const video=new Video(),seen=[];const player=new EpisodePlayback(video,{onFrame:i=>seen.push(i)});player.setEpisode(descriptor);video.ready();
 player.seekFrame(1000);video.dispatchEvent(new Event('seeked'));assert.equal(seen.at(-1),59);assert.equal(video.paused,true);
 await player.toggle();assert.ok(video.currentTime<1/30);player.setRate(2);assert.equal(video.playbackRate,2);player.destroy();
});
test('changing episode clears readiness and cannot play stale data',async()=>{
 const video=new Video(),player=new EpisodePlayback(video);player.setEpisode(descriptor);video.ready();await player.toggle();
 player.setEpisode({...descriptor,url:'second.mp4'});assert.equal(video.paused,true);assert.equal(await player.toggle(),false);
 assert.equal(video.callbacks.size,0);player.destroy();
});
test('decode and play errors stop playback and surface the error',async()=>{
 const video=new Video(),errors=[],player=new EpisodePlayback(video,{onError:e=>errors.push(e)});player.setEpisode(descriptor);video.ready();
 video.play=()=>Promise.reject(new Error('Blocked'));
 assert.equal(await player.toggle(),false);assert.match(errors.at(-1),/Blocked/);
 video.dispatchEvent(new Event('error'));assert.equal(player.ready,false);player.destroy();
});
