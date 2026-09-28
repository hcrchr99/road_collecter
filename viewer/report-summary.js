(function(root){
  'use strict';
  const numeric=v=>v!==null&&v!==undefined&&String(v).trim()!==''&&Number.isFinite(Number(v));
  function summarize(pack,vision,segment=null){
    if(!pack)throw Error('请先导入采集包');
    const events=pack.events.filter(e=>!segment?true:numeric(e.segment_id)?String(e.segment_id)===String(segment.segment_id):numeric(e.t_ms)&&numeric(segment.t_start_ms)&&numeric(segment.t_end_ms)&&+e.t_ms>=+segment.t_start_ms&&+e.t_ms<=+segment.t_end_ms);
    const ids=new Set(events.map(e=>String(e.id)));
    const reviews=(vision?.reviews||[]).filter(r=>r.pack_id===pack.packId&&ids.has(String(r.event_id)));
    const counts=list=>{const m=new Map();for(const label of list)m.set(label,(m.get(label)||0)+1);return [...m].sort((a,b)=>b[1]-a[1]);};
    const attention=reviews.filter(r=>r.needs_attention||r.confidence==='low');
    const disagree=reviews.filter(r=>r.agree_with_imu===false);
    return {packId:pack.packId,scope:segment?'路段 '+segment.segment_id:'全包汇总（非自动路段划分）',events,reviews,unreviewed:events.length-reviews.length,labels:counts(reviews.map(r=>r.visual_label)),algorithm:counts(events.map(e=>e.auto_label||'未记录')),attention,disagree,lowSpeed:events.filter(e=>String(e.low_speed)==='1').length,lowSpeedUnknown:events.filter(e=>!['0','1'].includes(String(e.low_speed))).length,example:vision?.generated==='human_example'||pack.meta?.demo===true};
  }
  if(typeof module!=='undefined'&&module.exports)module.exports={summarize};else root.RoadCheckReport={summarize};
})(typeof window==='undefined'?{}:window);
