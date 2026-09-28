function renderSummary(){
  const pack=state.pack;
  if(!pack){$('segmentTabs').replaceChildren();$('segmentFacts').textContent='等待采集包';$('reportBody').innerHTML='<div class="empty">导入采集包后自动生成事实汇总；导入 Agent 判读后自动补充视觉统计。</div>';return;}
  const segments=pack.segments||[];
  const seg=segments.find(s=>String(s.segment_id)===String(state.segment));
  $('segmentTabs').innerHTML='<button data-segment="all" class="'+(!seg?'active':'')+'">全包汇总</button>'+segments.map(s=>'<button data-segment="'+esc(s.segment_id)+'" class="'+(s===seg?'active':'')+'">路段 '+esc(s.segment_id)+'</button>').join('');
  const data=window.RoadCheckReport.summarize(pack,state.vision,seg);
  $('segmentFacts').textContent=seg?'范围按事件 segment_id 关联；缺少编号时按路段时间窗关联。未匹配事件仍保留在全包汇总中。':'当前显示整个采集包，不把全包冒充单一路段。'+(!segments.length?' 本包没有路段划分记录。':'');
  const dist=rows=>rows.length?rows.map(([label,count])=>esc(label)+' '+count+' 次').join(' · '):'暂无记录';
  const evidence=rows=>rows.map(r=>'<button data-event="'+esc(r.event_id??r.id)+'">事件 #'+esc(r.event_id??r.id)+'</button>').join(' ');
  const suggestions=[];
  if(data.attention.length)suggestions.push('优先人工核对 '+data.attention.length+' 条低置信或标记需处理的判读，不能直接当作确认病害。');
  if(data.disagree.length)suggestions.push('复核 '+data.disagree.length+' 条算法与视觉不一致的记录，结合图片和波形判断。');
  if(data.unreviewed)suggestions.push('还有 '+data.unreviewed+' 个事件没有视觉判读；请运行对应分析并导入结果，不能将未覆盖事件视为正常。');
  if(!suggestions.length)suggestions.push('当前记录没有上述待处理标记，仍需抽查现场证据；不代表道路无病害。');
  $('reportBody').innerHTML='<article class="card"><span class="tag '+(data.example?'example':'blue')+'">'+(data.example?'合成示例 · 非真实分析':'程序汇总 · 基于采集数据与导入判读')+'</span><h3 class="report-title">'+esc(data.scope)+'</h3><p>记录事件 '+data.events.length+' 个；已覆盖视觉判读 '+data.reviews.length+' 个；未覆盖 '+data.unreviewed+' 个。</p><p>低速记录 '+data.lowSpeed+' 个；低速状态未知 '+data.lowSpeedUnknown+' 个。</p><h3>视觉判读分布</h3><p>'+dist(data.labels)+'</p><h3>传感器算法标签分布</h3><p>'+dist(data.algorithm)+'</p><h3>核验与后续行动</h3><ul>'+suggestions.map(s=>'<li>'+esc(s)+'</li>').join('')+'</ul><p class="small muted">待处理数基于已导入判读的低置信或标记；失败、前置拦截及仅存在于 Agent 工作台的记录不计入视觉判读覆盖。待处理与分歧可能重叠，不可相加。</p><details><summary>查看需处理事件（'+data.attention.length+'）</summary><div class="row">'+evidence(data.attention)+'</div></details><details><summary>查看算法与视觉分歧（'+data.disagree.length+'）</summary><div class="row">'+evidence(data.disagree)+'</div></details><details><summary>查看本范围全部事件（'+data.events.length+'）</summary><div class="row">'+evidence(data.events)+'</div></details><p class="notice warn">本报告为确定性程序汇总，不是模型撰写的综合诊断。不生成 PCI / IRI、道路健康评分或维修优先级；没有跨趟对照，不推断复现率或病害发展趋势。具体养护决策需现场核验。</p></article>';
}
