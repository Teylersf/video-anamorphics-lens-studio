import type {Settings} from './settings'
export const base=(import.meta.env.VITE_API_BASE||'').replace(/\/$/,'')
export const url=(p?:string|null)=>!p?'':/^(https?:|blob:|data:)/.test(p)?p:`${base}${p.startsWith('/')?p:`/${p}`}`
export async function api<T>(path:string,init:RequestInit={}):Promise<T>{
 const response=await fetch(`${base}${path}`,init)
 if(!response.ok){let text=`${response.status} ${response.statusText}`;try{const d=await response.json();text=typeof d.detail==='string'?d.detail:JSON.stringify(d.detail||d.error||d)}catch{}throw new Error(text)}
 return response.json() as Promise<T>
}
export const post=<T,>(path:string,body:unknown={},signal?:AbortSignal)=>api<T>(path,{method:'POST',headers:{'Content-Type':'application/json','X-Studio-Client':'anamorphic-studio'},body:JSON.stringify(body),signal})
export type Media={id:string;name:string;width:number;height:number;fps:number;fps_exact:string;duration:number;codec:string;container:string;pixel_format:string;bit_depth:number;audio_streams:number|unknown[];transfer:string;preview_url:string;poster_url:string;warnings:string[]}
export type Status={ready:boolean;gpu:string|{name?:string;device?:string};ffmpeg:unknown;version:string}
export type Preview={original_url:string;processed_url:string;time:number;width:number;height:number;render_ms:number}
export type RenderResult={width:number;height:number;frames:number;fps_exact:string;codec:string;encoder:string;pixel_format:string;bit_depth:number;container:string;audio_streams:number;audio:string;encode_quality:string;transfer:string;warnings:string[];bytes?:number}
export type Job={id:string;status:string;progress:number;message?:string;output_url?:string;output_name?:string;error?:string;result?:RenderResult}
export type Config={defaults:Settings;profiles:Record<string,Settings>}
export const success=(j:Job)=>['done','complete','completed','succeeded','success'].includes(j.status)
export const terminal=(j:Job)=>success(j)||['failed','error','cancelled','canceled'].includes(j.status)
