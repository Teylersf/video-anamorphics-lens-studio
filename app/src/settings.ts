export const controls = {
 amount:{label:'Overall intensity',min:0,max:1,step:.01,value:.65,unit:'%'},
 flare:{label:'Flare intensity',min:0,max:2,step:.01,value:.55,unit:'%'},
 flare_length:{label:'Streak length',min:.02,max:.5,step:.005,value:.18,unit:'%'},
 threshold:{label:'Highlight threshold',min:0,max:1,step:.01,value:.8,unit:'%'},
 tint:{label:'Flare color',min:0,max:1,step:.01,value:0,unit:'%'},
 bloom:{label:'Bloom',min:0,max:1,step:.01,value:.15,unit:'%'},
 halation:{label:'Halation',min:0,max:1,step:.01,value:.1,unit:'%'},
 ghosts:{label:'Lens ghosts',min:0,max:1,step:.01,value:.08,unit:'%'},
 chromatic:{label:'Chromatic separation',min:0,max:1,step:.01,value:.12,unit:'%'},
 distortion:{label:'Barrel distortion',min:-1,max:1,step:.01,value:.05,unit:'%'},
 vignette:{label:'Vignette',min:0,max:1,step:.01,value:.15,unit:'%'},
 grain:{label:'Film grain',min:0,max:1,step:.01,value:.08,unit:'%'},
 bokeh:{label:'Focus falloff',min:0,max:1,step:.01,value:0,unit:'%'},
 ovality:{label:'Bokeh ovality',min:1,max:3,step:.05,value:1.8,unit:'×'},
 focus_y:{label:'Focus position',min:0,max:1,step:.01,value:.5,unit:'%'},
 focus_width:{label:'Focus band',min:0,max:1,step:.01,value:.6,unit:'%'},
 exposure:{label:'Exposure',min:-3,max:3,step:.05,value:0,unit:'EV'},
 contrast:{label:'Contrast',min:.5,max:1.5,step:.01,value:1,unit:'×'},
 saturation:{label:'Saturation',min:0,max:2,step:.01,value:1,unit:'×'},
} as const
export type Key=keyof typeof controls
export type Settings=Record<Key,number>
export type Profile='modern'|'vintage'|'clean'|'off'
export const keys=Object.keys(controls) as Key[]
export const defaults=Object.fromEntries(keys.map(k=>[k,controls[k].value])) as Settings
export const profiles:Record<Profile,{name:string;detail:string}>={modern:{name:'Modern',detail:'Cool · precise'},vintage:{name:'Vintage',detail:'Warm · organic'},clean:{name:'Clean',detail:'Subtle · refined'},off:{name:'Bypass',detail:'Original image'}}
export const fallbackLooks:Record<Profile,Settings>={modern:{...defaults},vintage:{...defaults,tint:.8,bloom:.28,halation:.25,grain:.18},clean:{...defaults,amount:.35,flare:.25,bloom:.06,ghosts:0,grain:.02},off:{...defaults,amount:0}}
export function validate(value:unknown,base:Settings=defaults):Settings{
 if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('Settings must be a JSON object.')
 const next={...base}
 for(const [key,n]of Object.entries(value)){
  if(!Object.hasOwn(controls,key))throw new Error(`Unknown setting: ${key}`)
  const r=controls[key as Key]
  if(typeof n!=='number'||!Number.isFinite(n)||n<r.min||n>r.max)throw new Error(`${r.label} must be between ${r.min} and ${r.max}.`)
  next[key as Key]=n
 }
 return next
}
