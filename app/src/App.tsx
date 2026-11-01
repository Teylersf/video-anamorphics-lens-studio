import {useCallback,useEffect,useRef,useState} from 'react'
import type {ChangeEvent,CSSProperties,PointerEvent} from 'react'
import {Aperture,ArrowDownToLine,ArrowLeftRight,Check,ChevronDown,ChevronLeft,ChevronRight,CircleHelp,Clapperboard,Cpu,Download,Expand,Film,FolderOpen,Gauge,GripVertical,HardDrive,Layers3,LoaderCircle,Maximize2,Pause,Play,RefreshCw,RotateCcw,Save,Settings2,ShieldCheck,SlidersHorizontal,Sparkles,Upload,X} from 'lucide-react'
import {api,base,post,success,terminal,url} from './api'
import type {Config,Job,Media,Preview,Status} from './api'
import {controls,defaults,fallbackLooks,profiles,validate} from './settings'
import type {Key,Profile,Settings} from './settings'

type View='wipe'|'original'|'processed'
type Mode='match'|'hevc'|'prores'|'lossless'
const modes:Record<Mode,{name:string;note:string}>={
 match:{name:'Match source',note:'Keep source container, codec, dimensions, frame rate and audio where supported. Codec lossless mode is used when available; source chroma is preserved. This is not a lossless RGB master.'},
 hevc:{name:'HEVC / H.265',note:'High-quality GPU encoding for efficient delivery. The engine reports the actual output settings; this is not a lossless RGB master.'},
 prores:{name:'ProRes master',note:'A high-quality editing master with larger files. ProRes encoding is not mathematically lossless.'},
 lossless:{name:'Lossless RGB master',note:'Preserve the processed RGB frames with a lossless codec. This does not undo source compression or changes made by the look. Lossless files can be substantially larger.'},
}
const groups:Array<{title:string;keys:Key[];open?:boolean;note?:string}>=[
 {title:'Lens character',keys:['flare','flare_length','threshold','tint','chromatic','distortion','vignette'],open:true},
 {title:'Light & texture',keys:['bloom','halation','ghosts','grain']},
 {title:'Focus & bokeh',keys:['bokeh','ovality','focus_y','focus_width'],note:'A shaped focus-band approximation, not estimated scene depth.'},
 {title:'Color adjustments',keys:['exposure','contrast','saturation']},
]
const timeLabel=(n:number)=>`${String(Math.floor(Math.max(0,n)/60)).padStart(2,'0')}:${(Math.max(0,n)%60).toFixed(3).padStart(6,'0')}`
const message=(e:unknown)=>e instanceof Error?e.message:'The operation could not finish.'
function Slider({name,value,onChange}:{name:Key;value:number;onChange:(k:Key,n:number)=>void}){
 const r=controls[name]
 const label=name==='tint'?(value<.01?'Blue':value>.99?'Amber':`${Math.round(value*100)}% warm`):r.unit==='%'?`${Math.round(value*100)}%`:r.unit==='EV'?`${value>0?'+':''}${value.toFixed(2)} EV`:`${value.toFixed(2)}×`
 return <div className={`parameter ${name==='tint'?'tint-control':''}`}><div className="parameter-label"><label htmlFor={`setting-${name}`}>{r.label}</label><output htmlFor={`setting-${name}`}>{label}</output></div><input id={`setting-${name}`} type="range" min={r.min} max={r.max} step={r.step} value={value} onChange={e=>onChange(name,Number(e.target.value))} style={{'--fill':`${(value-r.min)/(r.max-r.min)*100}%`} as CSSProperties}/>{name==='tint'&&<div className="range-ends"><span>Blue</span><span>Amber</span></div>}</div>
}

export default function App(){
 const [status,setStatus]=useState<Status|null>(null),[offline,setOffline]=useState(false)
 const [settings,setSettings]=useState<Settings>({...defaults}),[engineDefaults,setEngineDefaults]=useState({...defaults}),[looks,setLooks]=useState({...fallbackLooks}),[profile,setProfile]=useState<string>('modern')
 const [media,setMedia]=useState<Media|null>(null),[uploading,setUploading]=useState(false),[uploadPercent,setUploadPercent]=useState(0),[dragging,setDragging]=useState(false)
 const [time,setTime]=useState(0),[preview,setPreview]=useState<Preview|null>(null),[previewBusy,setPreviewBusy]=useState(false),[previewError,setPreviewError]=useState(''),[retry,setRetry]=useState(0)
 const [view,setView]=useState<View>('wipe'),[wipe,setWipe]=useState(50),[mode,setMode]=useState<Mode>('match')
 const [job,setJob]=useState<Job|null>(null),[isPreviewJob,setIsPreviewJob]=useState(false),[starting,setStarting]=useState(false),[cancelling,setCancelling]=useState(false)
 const [clip,setClip]=useState<Job|null>(null),[exported,setExported]=useState<Job|null>(null),[showClip,setShowClip]=useState(false),[playing,setPlaying]=useState(false)
 const [error,setError]=useState(''),[notice,setNotice]=useState('')
 const fileInput=useRef<HTMLInputElement>(null),lookInput=useRef<HTMLInputElement>(null),viewport=useRef<HTMLDivElement>(null),player=useRef<HTMLVideoElement>(null),xhr=useRef<XMLHttpRequest|null>(null),dirty=useRef(false),handled=useRef('')
 const busy=!!job&&!terminal(job),locked=uploading||busy||starting,maxTime=media?Math.max(0,media.duration-1/(media.fps||30)):0

 const operationLocked=useRef(locked)
 useEffect(()=>{operationLocked.current=locked},[locked])
 const connect=useCallback(async()=>{
  const [health,config]=await Promise.allSettled([api<Status>('/api/status'),api<Config>('/api/config')])
  if(health.status==='fulfilled'){setStatus(health.value);setOffline(false)}else{setStatus(null);setOffline(true)}
  if(config.status==='fulfilled')try{const d=validate(config.value.defaults||{}),p={...fallbackLooks};for(const k of Object.keys(profiles) as Profile[])if(config.value.profiles?.[k])p[k]=validate(config.value.profiles[k],d);setEngineDefaults(d);setLooks(p);if(!dirty.current)setSettings(d)}catch(e){setError(`Configuration: ${message(e)}`)}
 },[])
 useEffect(()=>{const initial=window.setTimeout(()=>void connect(),0);const timer=window.setInterval(()=>{api<Status>('/api/status').then(s=>{setStatus(s);setOffline(false)}).catch(()=>{setStatus(null);setOffline(true)})},15000);return()=>{window.clearTimeout(initial);window.clearInterval(timer);xhr.current?.abort()}},[connect])
 useEffect(()=>{if(!notice)return;const timer=window.setTimeout(()=>setNotice(''),4500);return()=>window.clearTimeout(timer)},[notice])
 useEffect(()=>{
  if(!media||!status?.ready||busy||starting)return
  const controller=new AbortController();let active=true
  const timer=window.setTimeout(()=>{setPreviewBusy(true);setPreviewError('');post<Preview>('/api/preview',{media_id:media.id,time,settings,width:1280},controller.signal).then(p=>{if(active)setPreview(p)}).catch(e=>{if(active&&!controller.signal.aborted)setPreviewError(message(e))}).finally(()=>{if(active)setPreviewBusy(false)})},350)
  return()=>{active=false;window.clearTimeout(timer);controller.abort()}
 },[media,time,settings,status?.ready,retry,busy,starting])
 const acceptJob=useCallback((next:Job,previewJob:boolean)=>{
  setJob(next)
  if(!terminal(next)||handled.current===next.id)return
  handled.current=next.id;setCancelling(false)
  if(success(next)&&next.output_url){if(previewJob){setClip(next);setShowClip(true);setNotice('Rendered preview is ready.')}else{setExported(next);setNotice('Your export is ready to download.')}}
  else if(['failed','error'].includes(next.status))setError(next.error||next.message||'The render failed.')
  else if(success(next))setError('The render finished without an output URL.')
  else setNotice('Render canceled.')
 },[])
 const jobId=job?.id
 useEffect(()=>{
  if(!jobId||!busy)return
  const controller=new AbortController();let timer:number|undefined
  const poll=async()=>{try{const next=await api<Job>(`/api/jobs/${encodeURIComponent(jobId)}`,{signal:controller.signal});if(controller.signal.aborted)return;acceptJob(next,isPreviewJob);if(!terminal(next))timer=window.setTimeout(poll,800)}catch(e){if(!controller.signal.aborted){setError(`Render status: ${message(e)}`);timer=window.setTimeout(poll,2000)}}}
  timer=window.setTimeout(poll,400)
  return()=>{controller.abort();window.clearTimeout(timer)}
 },[jobId,busy,isPreviewJob,acceptJob])

 function update(k:Key,n:number){if(locked)return;setClip(null);dirty.current=true;setSettings(s=>({...s,[k]:n}));setProfile('custom');setShowClip(false)}
 function choose(p:Profile){if(locked)return;setClip(null);dirty.current=true;setSettings({...looks[p]});setProfile(p);setShowClip(false)}
 function reset(){if(locked)return;setClip(null);dirty.current=true;setSettings({...engineDefaults});setProfile('modern');setShowClip(false);setNotice('Look reset to engine defaults.')}
 function seek(n:number){setTime(Math.max(0,Math.min(maxTime,Number.isFinite(n)?n:0)));setShowClip(false);setPlaying(false)}
 function upload(file?:File){
  if(!file)return
  if(locked){setError('Finish or cancel the current operation before importing another file.');return}
  setError('');setUploading(true);setUploadPercent(0)
  const x=new XMLHttpRequest();xhr.current=x;x.open('POST',`${base}/api/media`);x.setRequestHeader('X-Studio-Client','anamorphic-studio')
  x.upload.onprogress=e=>{if(e.lengthComputable)setUploadPercent(Math.round(e.loaded/e.total*100))}
  x.onload=()=>{setUploading(false);xhr.current=null;try{const d=JSON.parse(x.responseText);if(x.status<200||x.status>=300)throw new Error(typeof d.detail==='string'?d.detail:JSON.stringify(d.detail||d.error||d));if(!d.id||!d.width||!d.height)throw new Error('Incomplete media information returned.');setMedia(d);setTime(0);setPreview(null);setPreviewError('');setClip(null);setExported(null);setShowClip(false);setJob(null);setNotice('Footage imported. Preparing the first frame.')}catch(e){setError(message(e))}}
  x.onerror=()=>{setUploading(false);xhr.current=null;setError('Import connection failed. Check the local engine.')}
  x.onabort=()=>{setUploading(false);xhr.current=null;setNotice('Import canceled.')}
  const data=new FormData();data.append('file',file);x.send(data)
 }
 async function render(previewJob:boolean){if(!media||locked||!status?.ready)return;setError('');setStarting(true);setIsPreviewJob(previewJob);try{const next=await post<Job>('/api/jobs',{media_id:media.id,settings,mode,preview:previewJob});handled.current='';acceptJob(next,previewJob)}catch(e){setError(message(e))}finally{setStarting(false)}}
 async function cancel(){if(!job||!busy||cancelling)return;setCancelling(true);try{acceptJob(await post<Job>(`/api/jobs/${encodeURIComponent(job.id)}/cancel`),isPreviewJob)}catch(e){setError(message(e));setCancelling(false)}}
 function save(){const blob=new Blob([JSON.stringify({schema:1,app:'anamorphic-studio',settings},null,2)],{type:'application/json'}),href=URL.createObjectURL(blob),a=document.createElement('a');a.href=href;a.download=`anamorphic-${profile}.json`;a.click();window.setTimeout(()=>URL.revokeObjectURL(href),1000);setNotice('Look settings saved.')}
 async function load(e:ChangeEvent<HTMLInputElement>){const file=e.target.files?.[0];e.target.value='';if(!file||locked)return;try{if(file.size>65536)throw new Error('Look files must be smaller than 64 KiB.');const d=JSON.parse(await file.text());if(operationLocked.current)return;setClip(null);dirty.current=true;setSettings(validate(d.settings??d,engineDefaults));setProfile('custom');setShowClip(false);setNotice('Look settings loaded.')}catch(failure){setError(`Could not load look: ${message(failure)}`)}}
 function moveWipe(e:PointerEvent<HTMLDivElement>){if(e.buttons!==1||view!=='wipe'||showClip||!preview)return;e.currentTarget.setPointerCapture(e.pointerId);const b=e.currentTarget.getBoundingClientRect();setWipe(Math.max(0,Math.min(100,(e.clientX-b.left)/b.width*100)))}
 function playback(){if(!clip?.output_url){void render(true);return}if(!showClip){setShowClip(true);return}if(player.current?.paused)void player.current.play().catch(e=>setError(message(e)));else player.current?.pause()}
 const gpu=typeof status?.gpu==='string'?status.gpu:status?.gpu?.name||status?.gpu?.device||'GPU engine'
 const progress=job?Math.min(100,Math.max(0,(job.progress<=1?job.progress*100:job.progress)||0)):0
 const audio=media?(Array.isArray(media.audio_streams)?media.audio_streams.length:media.audio_streams):0
 const original=preview?url(preview.original_url):url(media?.poster_url)

 return <div className="studio">
  <input ref={fileInput} className="sr-only" type="file" accept="video/*,.mov,.mkv,.mxf,.mts,.m2ts,.avi,.webm" onChange={e=>{upload(e.target.files?.[0]);e.target.value=''}}/>
  <input ref={lookInput} className="sr-only" type="file" accept="application/json,.json" onChange={e=>void load(e)}/>
  <header className="topbar"><a className="brand" href="#workspace"><span className="brand-mark"><Aperture size={26} strokeWidth={1.3}/></span><span>ANAMORPHIC<small>STUDIO <i>/</i> CINEMA LAB</small></span></a><div className={`engine ${status?.ready?'connected':''}`} title={gpu}><i className="dot"/><span>{status?.ready?gpu:'Engine disconnected'}</span><b>LOCAL</b></div><div className="top-actions"><button disabled={locked} onClick={()=>lookInput.current?.click()} title="Load settings JSON"><FolderOpen size={15}/><span>Load look</span></button><button onClick={save} title="Save settings JSON"><Save size={15}/><span>Save look</span></button><i/><button className="icon-button" disabled={locked} onClick={reset} title="Reset all adjustments" aria-label="Reset all adjustments"><RotateCcw size={16}/></button></div></header>
  <main className="workspace" id="workspace">
   <aside className="panel lens-panel"><div className="panel-heading"><div><span className="eyebrow">THE LOOK</span><h2>Lens character</h2></div><SlidersHorizontal size={17}/></div><fieldset className="panel-body" disabled={locked}><div className="section-label">LENS PROFILE<span>{profile==='custom'?'CUSTOM':'4 LOOKS'}</span></div><div className="profiles">{(Object.keys(profiles) as Profile[]).map(p=><button key={p} className={`profile ${p} ${profile===p?'selected':''}`} onClick={()=>choose(p)} aria-pressed={profile===p}><span className="profile-art"><i/><b/>{profile===p&&<Check size={12}/>}</span><strong>{profiles[p].name}</strong><small>{profiles[p].detail}</small></button>)}</div><div className="master"><Slider name="amount" value={settings.amount} onChange={update}/></div>{groups.map(g=><details key={g.title} className="control-group" open={g.open}><summary>{g.title}<ChevronDown size={14}/></summary><div className="group-body">{g.keys.map(k=><Slider key={k} name={k} value={settings[k]} onChange={update}/>)}{g.note&&<p className="control-note"><CircleHelp size={12}/>{g.note}</p>}</div></details>)}<button className="reset-button" onClick={reset}><RotateCcw size={12}/>Reset all adjustments</button></fieldset><div className="panel-note"><ShieldCheck size={13}/>Your footage stays on this computer.</div></aside>

   <section className="viewer-panel"><div className="viewer-heading"><div><span className="eyebrow">ANAMORPHIC WORKSPACE</span><h1>{media?media.name:'A different kind of atmosphere.'}</h1></div><span className="look-badge"><Sparkles size={12}/>{profile==='custom'?'Custom look':profiles[profile as Profile]?.name||'Modern'}</span></div>
    {offline&&<div className="connection-notice"><Cpu size={15}/><span>The local engine is disconnected. Start it, then reconnect.</span><button onClick={()=>void connect()}>Reconnect<RefreshCw size={12}/></button></div>}
    <div className="viewer-toolbar"><span className="viewer-label"><i/>{showClip?'RENDERED PREVIEW':'FRAME PREVIEW'}<small>{showClip?'First 3 seconds':preview?`${preview.width} × ${preview.height}`:media?'Source poster':'GPU workspace'}</small></span><div className="comparison" aria-label="Comparison mode">{(['original','wipe','processed'] as View[]).map(v=><button key={v} onClick={()=>{setView(v);setShowClip(false)}} className={view===v&&!showClip?'active':''} aria-pressed={view===v&&!showClip}>{v==='wipe'?<><ArrowLeftRight size={12}/>Compare</>:v==='original'?'Before':'After'}</button>)}</div><button className="icon-button" title="Fullscreen viewer" aria-label="Fullscreen viewer" onClick={()=>{if(document.fullscreenElement)void document.exitFullscreen().catch(e=>setError(message(e)));else void viewport.current?.requestFullscreen().catch(e=>setError(message(e)))}}><Expand size={15}/></button></div>
    <div ref={viewport} className={`viewport ${!media?'empty':''}`} onPointerDown={moveWipe} onPointerMove={moveWipe} onDragOver={e=>{e.preventDefault();setDragging(true)}} onDragLeave={e=>{if(!e.currentTarget.contains(e.relatedTarget as Node))setDragging(false)}} onDrop={e=>{e.preventDefault();setDragging(false);upload(e.dataTransfer.files?.[0])}}>
     {!media&&!uploading&&<div className="empty-view"><i className="corner tl"/><i className="corner tr"/><i className="corner bl"/><i className="corner br"/><div className="empty-optic"><Aperture size={55} strokeWidth={.7}/></div><span className="eyebrow">LIGHT. TEXTURE. CHARACTER.</span><h2>Bring your footage<br/>into focus.</h2><p>Shape cinematic streaks, warm halation and<br/>lens character with a live GPU frame preview.</p><button className="primary import-main" onClick={()=>fileInput.current?.click()} disabled={locked}><Upload size={16}/>Import footage<span>＋</span></button><small>or drop a video here · original files stay untouched</small><div className="empty-features"><span><Cpu size={12}/>GPU processing</span><span><Maximize2 size={12}/>Source resolution</span><span><HardDrive size={12}/>Local workspace</span></div></div>}
     {media&&!showClip&&<><img className="frame" src={view==='processed'&&preview?url(preview.processed_url):original} alt={view==='processed'?'Processed video frame':'Original video frame'} draggable={false}/>{view==='wipe'&&preview&&<><img className="frame" src={url(preview.processed_url)} alt="Processed comparison frame" draggable={false} style={{clipPath:`inset(0 ${100-wipe}% 0 0)`}}/><div className="wipe-line" style={{left:`${wipe}%`}}><span><GripVertical size={18}/></span></div><input className="wipe-slider" type="range" min="0" max="100" value={wipe} aria-label="Before and after comparison position" onChange={e=>setWipe(Number(e.target.value))}/></>}<div className="frame-labels"><span>{view==='original'||!preview?'ORIGINAL':'PROCESSED'}</span>{view==='wipe'&&preview&&<span>ORIGINAL</span>}</div><div className="frame-caption"><span>{media.width} × {media.height} SOURCE</span><span>{timeLabel(preview?.time??time)}</span></div></>}
     {media&&showClip&&clip?.output_url&&<video ref={player} className="clip-player" src={url(clip.output_url)} controls autoPlay muted playsInline onPlay={()=>setPlaying(true)} onPause={()=>setPlaying(false)} onEnded={()=>setPlaying(false)}/>}
     {dragging&&<div className="overlay drop-overlay"><Upload size={34}/><strong>Drop to {media?'replace':'import'} footage</strong></div>}
     {uploading&&<div className="overlay"><LoaderCircle size={28} className="spin"/><strong>{uploadPercent<100?`Importing footage · ${uploadPercent}%`:'Reading media & preparing preview'}</strong><span>Keeping your original file intact</span><div className="upload-progress"><i style={{width:`${uploadPercent}%`}}/></div><button onClick={()=>xhr.current?.abort()}>Cancel import</button></div>}
     {media&&previewBusy&&!showClip&&<div className="render-indicator"><LoaderCircle size={12} className="spin"/>Rendering GPU frame</div>}
     {previewError&&!showClip&&<div className="preview-error"><span>{previewError}</span><button onClick={()=>setRetry(n=>n+1)}>Retry preview</button></div>}
    </div>
    <div className="transport"><div className="transport-buttons"><button className="icon-button" aria-label="Previous frame" disabled={!media||locked} onClick={()=>seek(time-1/(media?.fps||30))}><ChevronLeft size={17}/></button><button className="play-button" aria-label={playing&&showClip?'Pause rendered preview':clip?'Play rendered preview':'Render and play preview'} disabled={!media||locked||!status?.ready} onClick={playback}>{starting?<LoaderCircle className="spin" size={15}/>:playing&&showClip?<Pause size={15} fill="currentColor"/>:<Play size={15} fill="currentColor"/>}</button><button className="icon-button" aria-label="Next frame" disabled={!media||locked} onClick={()=>seek(time+1/(media?.fps||30))}><ChevronRight size={17}/></button></div><span className="time-code">{timeLabel(time)}</span><input className="timeline" type="range" aria-label="Source timeline" min="0" max={maxTime||1} step={media?1/(media.fps||30):.001} value={time} disabled={!media||locked} onChange={e=>seek(Number(e.target.value))}/><span className="duration">{timeLabel(media?.duration||0)}</span><div className="time-input"><label htmlFor="preview-time">TIME</label><input id="preview-time" aria-label="Preview time in seconds" type="number" step="0.001" min="0" max={maxTime} value={Number(time.toFixed(3))} disabled={!media||locked} onChange={e=>seek(Number(e.target.value))}/><span>s</span></div></div>
    <div className="viewer-footer"><span><i className={`dot ${preview?'ready':''}`}/>{showClip?'First 3 seconds · rendered preview':preview?`GPU frame · ${Math.round(preview.render_ms)} ms · preview only`:'A precise frame preview. A new point of view.'}</span>{showClip?<button onClick={()=>{setShowClip(false);setPlaying(false)}}><Layers3 size={12}/>Return to frame</button>:<button disabled={!media||locked||!status?.ready} onClick={()=>void render(true)}><Clapperboard size={13}/>Preview first 3 seconds</button>}</div>
    <div className="workflow"><div><b>01</b><span><strong>Choose your footage</strong><small>Work with the original source</small></span></div><div><b>02</b><span><strong>Shape the light</strong><small>Preview an exact moment</small></span></div><div><b>03</b><span><strong>Make it cinematic</strong><small>Export at native dimensions</small></span></div></div>
   </section>

   <aside className="panel delivery-panel"><div className="panel-heading"><div><span className="eyebrow">YOUR FOOTAGE</span><h2>Source & delivery</h2></div><Film size={17}/></div><div className="panel-body delivery-body"><div className="section-label">SOURCE MEDIA</div>
    {media?<div className="source-card"><div className="thumbnail"><img src={url(media.poster_url)} alt="Imported footage poster"/><span>{media.width>=7680?'8K SOURCE':media.width>=3840?'4K+ SOURCE':'SOURCE'}</span></div><strong title={media.name}>{media.name}</strong><dl><dt>Resolution</dt><dd>{media.width} × {media.height}</dd><dt>Frame rate</dt><dd>{media.fps_exact||media.fps} fps</dd><dt>Duration</dt><dd>{timeLabel(media.duration)}</dd><dt>Codec</dt><dd>{media.codec?.toUpperCase()}</dd><dt>Pixel format</dt><dd>{media.pixel_format} · {media.bit_depth||8}-bit</dd><dt>Container</dt><dd>{media.container}</dd><dt>Transfer</dt><dd>{media.transfer||'Unspecified'}</dd><dt>Audio</dt><dd>{audio?`${audio} stream${audio===1?'':'s'}`:'No audio'}</dd></dl><button className="secondary" disabled={locked} onClick={()=>fileInput.current?.click()}><FolderOpen size={13}/>Replace footage</button></div>:<button className="source-empty" disabled={locked} onClick={()=>fileInput.current?.click()}><Film size={25} strokeWidth={1.2}/><strong>No source selected</strong><span>Import video to begin</span><small><Upload size={12}/>Browse files</small></button>}
    {!!media?.warnings?.length&&<div className="warnings">{media.warnings.map((w,i)=><p key={i}><CircleHelp size={12}/>{w}</p>)}</div>}
    <section className="export-settings"><div className="section-label">EXPORT SETTINGS<Settings2 size={12}/></div><label htmlFor="export-mode">Output format</label><div className="select-wrap"><select id="export-mode" value={mode} disabled={locked} onChange={e=>setMode(e.target.value as Mode)}>{(Object.keys(modes) as Mode[]).map(m=><option key={m} value={m}>{modes[m].name}</option>)}</select><ChevronDown size={14}/></div><p>{modes[mode].note}</p><div className="export-properties"><div><Maximize2 size={13}/><span>Resolution</span><b>{media?`${media.width} × ${media.height}`:'Match source'}</b></div><div><Gauge size={13}/><span>Frame rate</span><b>Preserve source</b></div><div><Layers3 size={13}/><span>Audio</span><b>{media&&!audio?'None in source':mode==='match'?'Copy original':'Copy where compatible'}</b></div></div><div className="quality-note"><ShieldCheck size={14}/><span>Source dimensions stay intact.<br/>Original files are never overwritten.</span></div></section>
    {job&&<div className="job-card" aria-live="polite"><div className="job-heading">{busy?<LoaderCircle size={14} className="spin"/>:success(job)?<Check size={14}/>:<CircleHelp size={14}/>}<strong>{busy?isPreviewJob?'Rendering preview':'Exporting footage':success(job)?'Render complete':['cancelled','canceled'].includes(job.status)?'Render canceled':'Render failed'}</strong><b>{Math.round(progress)}%</b></div><div className="job-progress"><i style={{width:`${success(job)?100:progress}%`}}/></div><p>{job.message||job.error||job.status}</p>{busy&&<button onClick={()=>void cancel()} disabled={cancelling}><X size={12}/>{cancelling?'Canceling…':'Cancel render'}</button>}</div>}
    {exported?.result&&<section className="result-details" aria-label="Verified export details"><div className="section-label">ACTUAL OUTPUT<Check size={12}/></div><dl><dt>Video</dt><dd>{exported.result.codec.toUpperCase()} · {exported.result.container.toUpperCase()}</dd><dt>Dimensions</dt><dd>{exported.result.width} × {exported.result.height}</dd><dt>Pixels</dt><dd>{exported.result.pixel_format} · {exported.result.bit_depth}-bit</dd><dt>Quality</dt><dd>{exported.result.encode_quality}</dd><dt>Audio</dt><dd>{exported.result.audio}</dd><dt>Frames</dt><dd>{exported.result.frames}</dd></dl>{exported.result.warnings?.map((w,i)=><p key={i}>{w}</p>)}</section>}
    {exported?.output_url&&<a className="download-card" href={url(exported.output_url)} download={exported.output_name||true}><span><Check size={14}/><strong>Export ready</strong><small>{exported.output_name||'Processed footage'}</small></span><Download size={19}/></a>}
   </div><div className="export-footer"><button className="primary export-button" onClick={()=>void render(false)} disabled={!media||locked||!status?.ready}>{starting&&!isPreviewJob?<LoaderCircle size={16} className="spin"/>:<ArrowDownToLine size={16}/>}<span>{busy&&!isPreviewJob?'Export in progress':'Export footage'}</span><ChevronRight size={15}/></button><small>Processed locally · GPU accelerated</small></div></aside>
  </main>
  <footer className="statusbar"><span><i className={`dot ${status?.ready?'ready':''}`}/>{status?.ready?'Engine ready':'Waiting for local engine'}<i>/</i>{status?.version?`v${status.version}`:'Anamorphic Studio'}</span><span>ORIGINAL RESOLUTION. YOUR CREATIVE DIRECTION.</span><span><HardDrive size={11}/>LOCAL WORKSPACE</span></footer>
  {(error||notice)&&<div className={`toast ${error?'error':''}`} role={error?'alert':'status'}>{error?<CircleHelp size={17}/>:<Check size={17}/>}<span>{error||notice}</span><button className="icon-button" aria-label="Dismiss notification" onClick={()=>{setError('');setNotice('')}}><X size={15}/></button></div>}
 </div>
}
