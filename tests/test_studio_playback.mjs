import test from 'node:test';
import assert from 'node:assert/strict';
import {Playback} from '../web/playback.mjs';
class Video extends EventTarget {
 constructor(){super();this.paused=true;this.readyState=0;this.currentTime=0;this.duration=8;this.ended=false;this.src='';this.loads=0;}
 load(){this.loads++;this.readyState=0;}
 pause(){this.paused=true;this.dispatchEvent(new Event('pause'));}
 play(){this.paused=false;this.dispatchEvent(new Event('play'));return Promise.resolve();}
 removeAttribute(){this.src='';}
 ready(){this.readyState=4;this.dispatchEvent(new Event('loadeddata'));}
}
test('same video does not restart when plan status is refreshed',()=>{const v=new Video(),p=new Playback(v);p.setSource('a.mp4');v.ready();v.currentTime=3;p.setSource('a.mp4');assert.equal(v.currentTime,3);assert.equal(v.loads,1);p.destroy();});
test('cannot play missing media; end replay resets and new source waits for ready',async()=>{const v=new Video(),p=new Playback(v);assert.equal(await p.toggle(),false);p.setSource('a.mp4');assert.equal(await p.toggle(),false);v.ready();v.ended=true;v.currentTime=8;await p.toggle();assert.equal(v.currentTime,0);assert.equal(v.paused,false);p.setSource('b.mp4');assert.equal(v.paused,true);assert.equal(await p.toggle(),false);p.destroy();});
test('timeupdate and final frame repaint both views even without animation callbacks',()=>{const v=new Video();let frames=0;const p=new Playback(v,{onFrame:()=>frames++});p.setSource('a.mp4');v.ready();const n=frames;v.currentTime=4;v.dispatchEvent(new Event('timeupdate'));v.dispatchEvent(new Event('ended'));assert.equal(frames,n+2);p.destroy();});
