// Minimal DOM harness for UI lifecycle; not a substitute for visual browser QA.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const api=require('../viewer/agent-log.js'),nodes=new Map(),listeners=new Map();
class Element{
  constructor(tag='div'){this.tag=tag;this.children=[];this._text='';this.value='';this.checked=false;this.hidden=false;}
  set innerHTML(html){for(const m of html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"[^>]*>/g)){const e=new Element(m[1]);e.id=m[2];e.hidden=m[0].includes(' hidden');nodes.set(e.id,e);}}
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(e=>e.textContent).join('');}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;this._text='';if(this.tag==='select')this.value=children.length?String(children[0].value):'';}
  before(node){nodes.set(node.id,node);}
  click(){this.clicked=true;if(this.onclick)this.onclick();}
  scrollIntoView(){this.scrolled=true;}
}
nodes.set('vision',new Element());nodes.set('explore',new Element());
const nav=new Element('nav'),document={createElement:tag=>new Element(tag),getElementById:id=>nodes.get(id),querySelector:()=>nav,addEventListener:(name,fn)=>listeners.set(name,fn)};
const state={pack:{packId:'pack',events:[{id:'0'},{id:'1'}]},demo:false};let selected;
const ctx=vm.createContext({document,window:{RoadCheckAgentLog:api},state,selectEvent:id=>{selected=id;}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../viewer/agent-panel.js'),'utf8'),ctx);
const get=id=>nodes.get(id),trace={event_id:0,rounds:[{round:0,status:'ok',label:'标线',conf_model:'low'}],escalations:[],final:'escalated'};
const file=(name,rows)=>({name,size:100,text:async()=>rows.map(JSON.stringify).join('\n')});
async function upload(files){const target={files,value:'chosen'};await get('agentFiles').onchange({target});assert.equal(target.value,'');}
async function main(){
  get('agentChoose').click();assert.match(get('agentStatus').textContent,/请填写/);
  get('agentPackId').value='pack';get('agentConfirm').checked=true;get('agentChoose').click();assert.equal(get('agentFiles').clicked,true);
  await upload([file('trace.jsonl',[trace]),file('reviews.jsonl',[{event_id:0,visual_label:'标线',source:'llm',confidence:'low',why:'<script>not executable</script>'}])]);
  assert.equal(get('agentResults').hidden,false);assert.match(get('agentDetail').textContent,/首轮判读/);assert.match(get('agentDetail').textContent,/<script>not executable<\/script>/);
  get('agentDetail').children.find(e=>e.tag==='button').click();assert.equal(selected,0);assert.equal(get('explore').scrolled,true);
  await upload([file('trace.jsonl',[{...trace,event_id:99}])]);assert.match(get('agentStatus').textContent,/导入失败/);assert.match(get('agentDetail').textContent,/事件 #0/);
  listeners.get('roadcheck:pack-change')();assert.equal(get('agentResults').hidden,false);
  state.pack={packId:'new',events:[]};listeners.get('roadcheck:pack-change')();assert.equal(get('agentResults').hidden,true);assert.equal(get('agentConfirm').checked,false);
  state.demo=true;get('agentChoose').click();assert.match(get('agentStatus').textContent,/真实采集包/);
  console.log('PASS Agent panel DOM lifecycle: confirmation, import, plain-text output, event linkage, failed import retention, pack change, demo guard');
}
main().catch(e=>{console.error(e);process.exitCode=1;});
