const assert=require('node:assert/strict');
const api=require('../viewer/agent-log.js');
const line=JSON.stringify;
const trace={event_id:0,rounds:[{round:0,status:'ok',label:'坑洼',conf_model:'low',actions:['r3_two_frames']},{round:1,status:'ok',label:'标线',conf_model:'mid'}],escalations:[],final:'converged'};
const review={event_id:0,visual_label:'标线',confidence:'high',why:'<img src=x onerror=alert(1)>',source:'llm',_agent:{calls:2}};
const input={'trace.jsonl':line(trace),'reviews.jsonl':line(review),'escalate.jsonl':''};
let data=api.join(input,'pack',[0,1]);assert.equal(data.events.length,1);assert.equal(data.events[0].review.visual_label,'标线');assert.equal(data.events[0].needsAttention,false);
assert.equal(data.events[0].review.why,review.why); // UI uses textContent, not HTML.
assert.equal(api.parse('\ufeff'+line(trace)+'\r\n','trace').length,1);
assert.throws(()=>api.join(input,'pack',[1]),/不存在/);
assert.throws(()=>api.join({'trace.jsonl':line({...trace,pack_id:'other'})},'pack',[0]),/不一致/);
assert.throws(()=>api.join({'trace.jsonl':line({...trace,rounds:null})},'pack',[0]),/rounds/);
assert.throws(()=>api.join({'trace.jsonl':'{"event_id":0}\nnope'},'pack',[0]),/第 2 行/);
assert.throws(()=>api.join({'trace.jsonl':line({...trace,event_id:'0'})},'pack',[0]),/event_id/);
assert.throws(()=>api.join({'trace.jsonl':''},'pack',[0]),/没有事件/);
data=api.join({'reviews.jsonl':line({...review,source:'net_failed'})},'pack',[0]);assert.equal(data.events[0].needsAttention,true);assert.ok(data.warnings.length);
data=api.join({...input,'trace.jsonl':line({...trace,final:'budget_exhausted'})},'pack',[0]);assert.equal(data.events[0].needsAttention,true);
data=api.join({...input,'trace.jsonl':line(trace)+'\n'+line(trace),'escalate.jsonl':line({event_id:0,reason:['低置信'],at:'2026-09-27'})},'pack',[0]);assert.equal(data.events[0].traces.length,2);assert.equal(data.events[0].needsAttention,true);assert.ok(data.warnings.length);
assert.throws(()=>api.join({'escalate.jsonl':line({event_id:0,reason:[]})},'pack',[0]),/请选择/);
const fs=require('node:fs'),vm=require('node:vm');new vm.Script(fs.readFileSync(require('node:path').join(__dirname,'../viewer/agent-panel.js'),'utf8'));
const converted=api.toVision(api.join(input,'pack',[0]));assert.equal(converted.reviews[0].pack_id,'pack');assert.equal(converted.reviews[0].visual_label,'标线');
for(const source of ['net_failed','parse_failed','l0_blocked']){const d=api.join({'reviews.jsonl':line(review)+'\n'+line({...review,source})},'pack',[0]);assert.equal(api.toVision(d).reviews.length,0);assert.equal(api.toVision(d).skipped,1);}
assert.throws(()=>api.toVision(api.join({'reviews.jsonl':line({...review,source:'unknown'})},'pack',[0])),/来源未知/);
console.log('PASS Agent logs: trace/review/escalation, 11-class compatibility, missing/duplicate records, source mismatch, malformed JSON and script syntax');
