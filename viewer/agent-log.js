/* Adapter for llm_dev agent JSONL. No network calls or storage. */
(function(root){
  'use strict';
  const endings={converged:'本轮自检通过',escalated:'已转人工复核',budget_exhausted:'预算耗尽 · 待处理',no_new_evidence:'无新增证据 · 待处理',blocked:'图片不可判读',net_failed:'模型请求失败',parse_failed:'模型结果解析失败'};
  const actions={r1_post_wave:'补充冲击后图片与波形',r2_more_frames:'补充并增强图片',r3_two_frames:'补充两张独立图片',r4_more_evidence:'补充更多图片'};
  function parse(text,name){
    if(text.length>10*1024*1024)throw Error(name+' 超过 10 MB，请分批导出');
    const lines=text.replace(/^\uFEFF/,'').split(/\r?\n/),out=[];
    lines.forEach((line,i)=>{if(!line.trim())return;let row;
      try{row=JSON.parse(line);}catch{throw Error(name+' 第 '+(i+1)+' 行不是有效 JSON');}
      if(!row||Array.isArray(row)||typeof row!=='object'||!Number.isInteger(row.event_id)||row.event_id<0)throw Error(name+' 第 '+(i+1)+' 行缺少有效 event_id');
      if(out.length>=20000)throw Error(name+' 记录过多');out.push(row);
    });return out;
  }
  function join(files,packId,eventIds){
    if(!Object.hasOwn(files,'trace.jsonl')&&!Object.hasOwn(files,'reviews.jsonl'))throw Error('请选择 trace.jsonl 或 reviews.jsonl，可同时选择 escalate.jsonl');
    const events=new Map(),warnings=[],valid=new Set(eventIds.map(String));
    for(const name of Object.keys(files)){
      if(!['trace.jsonl','reviews.jsonl','escalate.jsonl'].includes(name))throw Error('不支持的文件：'+name);
      for(const row of parse(files[name],name)){
        if(row.pack_id!==undefined&&row.pack_id!==packId)throw Error(name+' 的 pack_id 与当前采集包不一致');
        if(!valid.has(String(row.event_id)))throw Error(name+' 引用了当前采集包不存在的事件 #'+row.event_id);
        if(!events.has(row.event_id))events.set(row.event_id,{id:row.event_id,traces:[],reviews:[],escalations:[]});
        if(name==='trace.jsonl'){
          if(!Array.isArray(row.rounds)||row.rounds.length>100||!row.rounds.every(r=>r&&Number.isInteger(r.round)&&r.round>=0&&typeof r.status==='string'))throw Error('事件 #'+row.event_id+' 的 rounds 无效');
          if(row.escalations!==undefined&&(!Array.isArray(row.escalations)||!row.escalations.every(x=>typeof x==='string')))throw Error('escalations 必须为文本数组');
          for(const r of row.rounds)if(r.actions!==undefined&&(!Array.isArray(r.actions)||!r.actions.every(a=>typeof a==='string')))throw Error('actions 必须为文本数组');
          events.get(row.event_id).traces.push(row);
        }else if(name==='reviews.jsonl')events.get(row.event_id).reviews.push(row);
        else{
          if(!Array.isArray(row.reason)||!row.reason.every(x=>typeof x==='string'))throw Error('复核原因必须为文本数组');
          events.get(row.event_id).escalations.push(row);
        }
      }
    }
    if(!events.size)throw Error('文件没有事件记录');
    for(const e of events.values()){
      e.trace=e.traces.at(-1);e.review=e.reviews.at(-1);
      e.status=e.trace?.final||e.review?.source||'unknown';
      e.needsAttention=e.escalations.length>0||e.traces.some(t=>t.escalations?.length)||['escalated','budget_exhausted','no_new_evidence','net_failed','parse_failed'].includes(e.status)||['net_failed','parse_failed'].includes(e.review?.source);
      if(e.traces.length>1||e.reviews.length>1)warnings.push('事件 #'+e.id+' 有重复执行记录；显示最后一条结果，保留全部轨迹，无法确认它们属于同一次运行。');
      if(!e.trace)warnings.push('事件 #'+e.id+' 缺少过程轨迹，不能推断执行步骤。');
    }
    return {packId,events:[...events.values()].sort((a,b)=>a.id-b.id),warnings};
  }
  const api={parse,join,endings,actions};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.RoadCheckAgentLog=api;
})(typeof window==='undefined'?{}:window);
