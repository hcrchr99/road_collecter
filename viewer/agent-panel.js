/* Classic script: consumes the viewer's current pack and event selection. */
(()=>{
  'use strict';
  const el=id=>document.getElementById(id),api=window.RoadCheckAgentLog;
  const host=document.createElement('section');host.id='agent';host.className='section';
  host.innerHTML=`<div class="section-head"><div><h2>智能体工作台</h2><p class="small muted">查看已有 Agent 的逐轮判读和取证动作。</p></div><span class="tag blue">运行记录回放 · 非实时执行</span></div>
  <div class="card"><p>先导入采集包，再选择同一次 Agent 运行输出目录中的日志。文件仅在本页读取，不上传。</p>
  <p class="small muted">支持 trace.jsonl（过程）、reviews.jsonl（结果）、escalate.jsonl（人工复核）。可一次选择多个文件。旧日志没有采集包编号和运行编号，请人工确认来源，不能仅凭事件编号判断属于同一包。</p>
  <div class="row"><label>日志所属采集包编号 <input id="agentPackId" placeholder="ZIP 文件名，不含 .zip" autocomplete="off"></label><label class="small"><input id="agentConfirm" type="checkbox">我已核对日志来自当前采集包及同一次运行</label><button id="agentChoose" class="primary">导入 Agent 日志</button><button id="agentClear">清除工作记录</button><input id="agentFiles" type="file" accept=".jsonl" multiple hidden></div>
  <p id="agentCurrentPack" class="small muted"></p><p id="agentStatus" class="small" role="status" aria-live="polite">尚未导入运行记录。不会自动运行模型或产生 API 费用。</p></div>
  <div id="agentResults" hidden><div id="agentSummary" class="strip"></div><div id="agentWarnings" class="notice warn" hidden></div><div class="agent-layout"><div class="card"><label>查看事件 <select id="agentEvent" aria-label="Agent 事件"></select></label><p><label class="small"><input id="agentAttention" type="checkbox">仅看需处理事件</label></p><p class="small muted">模型自报置信度与规则推导的最终置信度分别展示，不代表准确率。没有时间记录的步骤不会编造耗时。</p></div><div id="agentDetail" class="card"></div></div></div>`;
  document.getElementById('vision').before(host);
  const nav=document.querySelector('nav');if(nav){const link=document.createElement('a');link.href='#agent';link.textContent='智能体工作台';nav.append(link);}
  let data=null,boundPack=null,importToken=0;
  const text=v=>typeof v==='string'||typeof v==='number'?String(v):'未记录';
  const confidence=v=>({high:'高',mid:'中',low:'低'}[v]||'未记录');
  function reset(){data=null;boundPack=null;importToken++;el('agentResults').hidden=true;el('agentFiles').value='';el('agentConfirm').checked=false;el('agentStatus').textContent='尚未导入运行记录。不会自动运行模型或产生 API 费用。';el('agentAttention').checked=false;}
  let seenPack;
  function packChanged(){if(seenPack===state.pack)return;seenPack=state.pack;reset();el('agentPackId').value='';el('agentCurrentPack').textContent=state.pack?'当前采集包：'+state.pack.packId+(state.demo?'（合成演示；请导入真实包后加载真实日志）':''):'请先导入采集包。';}
  function renderList(){
    const rows=data.events.filter(e=>!el('agentAttention').checked||e.needsAttention);
    el('agentEvent').replaceChildren(...rows.map(e=>{const o=document.createElement('option');o.value=e.id;o.textContent='#'+e.id+' · '+(api.endings[e.status]||e.status)+(e.needsAttention?' · 需处理':'');return o;}));renderDetail();
  }
  function renderDetail(){
    const target=el('agentDetail');target.replaceChildren();
    const e=data?.events.find(x=>String(x.id)===el('agentEvent').value);if(!e){target.textContent='没有符合筛选条件的事件。';return;}
    function add(parent,tag,value,cls){const node=document.createElement(tag);node.textContent=value;if(cls)node.className=cls;parent.append(node);return node;}
    add(target,'h3','事件 #'+e.id+' · '+(api.endings[e.status]||text(e.status)));
    const button=add(target,'button','回看此事件的图片、波形和地图');button.onclick=()=>{if(boundPack!==state.pack)return;selectEvent(e.id);el('explore').scrollIntoView({behavior:'smooth'});};
    if(e.review){add(target,'p','最终结果：'+text(e.review.visual_label)+' ｜ 最终置信度：'+confidence(e.review.confidence));add(target,'p',text(e.review.why||e.review.error));add(target,'p','结果来源：'+text(e.review.source)+' ｜ 模型调用次数：'+text(e.review._agent?.calls),'small muted');}
    else add(target,'p','未导入最终结果文件；不能把某轮判读当成最终结论。','notice warn');
    if(e.traces.length===0)add(target,'p','缺少 trace.jsonl 中的对应记录，过程未知。','notice warn');
    e.traces.forEach((trace,index)=>{
      add(target,'h3','轨迹记录 '+(index+1)+' · '+(api.endings[trace.final]||text(trace.final)));
      if(!trace.rounds.length)add(target,'p',trace.final==='blocked'?'前置图片检查拦截，未调用模型。':'日志没有逐轮记录。');
      const list=document.createElement('ol');list.className='agent-timeline';target.append(list);
      trace.rounds.forEach(r=>{const li=document.createElement('li');list.append(li);add(li,'strong',(r.round===0?'首轮判读':'补充取证第 '+r.round+' 轮')+' · '+(r.status==='ok'?'判读返回':text(r.status)));add(li,'p',r.status==='ok'?'本轮分类：'+text(r.label)+' ｜ 模型自报置信度：'+confidence(r.conf_model):'失败原因：'+text(r.error));if(r.actions?.length)add(li,'p','后续取证动作：'+r.actions.map(a=>api.actions[a]||a).join('；'));});
      for(const reason of trace.escalations||[])add(target,'p','人工复核原因：'+reason,'notice warn');
    });
    for(const item of e.escalations)add(target,'p','复核队列：'+item.reason.join('；')+' ｜ 登记时间：'+text(item.at),'notice warn');
    if(['budget_exhausted','no_new_evidence'].includes(e.status))add(target,'p','此状态不代表分析通过。请由人工确认后续处理；原程序不一定已将它写入复核队列。','notice warn');
    add(target,'p','日志未提供逐步骤时间及完整证据文件索引；回看按钮展示包内事件资料，不宣称是本轮实际发送给模型的全部证据。','small muted');
  }
  el('agentChoose').onclick=()=>{if(!state.pack||state.demo){el('agentStatus').textContent='请先导入真实采集包，再关联 Agent 日志。';return;}if(!el('agentConfirm').checked||el('agentPackId').value.trim()!==state.pack.packId){el('agentStatus').textContent='请填写与当前包完全一致的编号，并确认日志来源。';return;}el('agentFiles').click();};
  el('agentFiles').onchange=async ev=>{
    const files=[...ev.target.files];if(!files.length)return;const pack=state.pack,token=++importToken;
    try{
      if(!pack||state.demo||!el('agentConfirm').checked||el('agentPackId').value.trim()!==pack.packId)throw Error('采集包或来源确认已变化，请重新选择');
      const inputs={};for(const file of files){if(!['trace.jsonl','reviews.jsonl','escalate.jsonl'].includes(file.name))throw Error('请选择原始日志文件 trace.jsonl、reviews.jsonl、escalate.jsonl');if(file.size>10*1024*1024)throw Error(file.name+' 超过 10 MB');if(Object.hasOwn(inputs,file.name))throw Error('同名日志只能选择一份');inputs[file.name]=await file.text();}
      if(token!==importToken||state.pack!==pack)return;
      const next=api.join(inputs,pack.packId,pack.events.map(e=>e.id));data=next;boundPack=pack;
      el('agentResults').hidden=false;el('agentSummary').textContent='已导入 '+data.events.length+' 个事件的记录 / 当前包 '+pack.events.length+' 个事件 · 需处理 '+data.events.filter(e=>e.needsAttention).length+' 个 · 关联方式：人工确认来源';
      el('agentWarnings').hidden=!data.warnings.length;el('agentWarnings').textContent=data.warnings.slice(0,10).join('\n')+(data.warnings.length>10?'\n另有 '+(data.warnings.length-10)+' 条记录提示':'');
      el('agentStatus').textContent='日志已载入，仅回放已记录的过程；未启动模型。';renderList();
    }catch(error){if(token===importToken)el('agentStatus').textContent='导入失败：'+error.message+'（原有记录未变）';}finally{ev.target.value='';}
  };
  el('agentClear').onclick=reset;el('agentEvent').onchange=renderDetail;el('agentAttention').onchange=renderList;
  document.addEventListener('roadcheck:pack-change',packChanged);packChanged();
})();
