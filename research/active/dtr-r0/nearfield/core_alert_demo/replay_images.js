/* Retain decoded original images; never interpolate or modify replay observations. */
(function(root){
  'use strict';
  class DecodedFrames {
    constructor({createImage=()=>new Image(),limit=32}={}) {
      this.createImage=createImage;this.limit=limit;this.cache=new Map();this.warmToken=0;
    }
    peek(url){return this.cache.get(url)?.ready||null;}
    get(url){
      let entry=this.cache.get(url);
      if(entry){this.cache.delete(url);this.cache.set(url,entry);return entry.promise;}
      const image=this.createImage();image.decoding='async';image.src=url;
      entry={ready:null,promise:null};
      entry.promise=image.decode().then(()=>{entry.ready=image;return image;}).catch(error=>{
        if(this.cache.get(url)===entry)this.cache.delete(url);
        throw error;
      });
      this.cache.set(url,entry);
      while(this.cache.size>this.limit)this.cache.delete(this.cache.keys().next().value);
      return entry.promise;
    }
    warm(urls){
      const token=++this.warmToken,queue=[...new Set(urls)].slice(0,this.limit);
      const worker=async()=>{
        while(token===this.warmToken&&queue.length){
          try{await this.get(queue.shift());}catch{/* A requested missing frame reports its own error. */}
        }
      };
      // Limit disk reads and decoding bursts while the visible frame has priority.
      return Promise.all([worker(),worker(),worker()]);
    }
  }
  class FramePresenter {
    constructor(images,{busy=()=>{},failure=()=>{},delay=180}={}) {
      this.images=images;this.busy=busy;this.failure=failure;this.delay=delay;this.token=0;this.timer=null;
    }
    cancel(){++this.token;clearTimeout(this.timer);this.busy(false);}
    async show(url,commit){
      this.cancel();const token=this.token;
      const ready=this.images.peek(url);
      if(ready){commit(ready);return true;}
      this.timer=setTimeout(()=>{if(token===this.token)this.busy(true);},this.delay);
      try{
        const image=await this.images.get(url);
        if(token!==this.token)return false;
        clearTimeout(this.timer);this.busy(false);commit(image);return true;
      }catch(error){
        if(token===this.token){clearTimeout(this.timer);this.busy(false);this.failure(error);}
        return false;
      }
    }
  }
  if(typeof module!=='undefined'&&module.exports)module.exports={DecodedFrames,FramePresenter};
  else root.ReplayImages={DecodedFrames,FramePresenter};
})(typeof window==='undefined'?{}:window);
