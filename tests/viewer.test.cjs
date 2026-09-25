/* Run: node tests/viewer.test.cjs. Fixtures are synthetic and kept in ignored userdata/. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const {performance} = require('node:perf_hooks');
const html = fs.readFileSync(path.join(__dirname, '../viewer/index.html'), 'utf8');
const ctx = vm.createContext({TextDecoder, Uint8Array, DataView});
vm.runInContext(html.match(/<script id="core">([\s\S]*?)<\/script>/)[1] + '\nthis.api={csvParse,zipIndex,parsePack,normalizePack,validateResult};', ctx);
const api = ctx.api;
const arrayBuffer = b => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
let passed = 0;
function test(name, fn) { fn(); passed++; console.log('PASS', name); }
function crc32(b) { let c=0xffffffff; for(const x of b){c^=x;for(let j=0;j<8;j++) c=(c>>>1)^((c&1)?0xedb88320:0);} return (c^0xffffffff)>>>0; }
function zip(entries, method = 0) {
  const local=[],central=[];let offset=0;
  for(const [filename, content] of Object.entries(entries)){
    const name=Buffer.from(filename),data=Buffer.isBuffer(content)?content:Buffer.from(content),crc=crc32(data);
    const h=Buffer.alloc(30);h.writeUInt32LE(0x04034b50);h.writeUInt16LE(20,4);h.writeUInt16LE(0x800,6);h.writeUInt16LE(method,8);h.writeUInt32LE(crc,14);h.writeUInt32LE(data.length,18);h.writeUInt32LE(data.length,22);h.writeUInt16LE(name.length,26);
    const c=Buffer.alloc(46);c.writeUInt32LE(0x02014b50);c.writeUInt16LE(20,4);c.writeUInt16LE(20,6);c.writeUInt16LE(0x800,8);c.writeUInt16LE(method,10);c.writeUInt32LE(crc,16);c.writeUInt32LE(data.length,20);c.writeUInt32LE(data.length,24);c.writeUInt16LE(name.length,28);c.writeUInt32LE(offset,42);
    local.push(h,name,data);central.push(c,name);offset+=h.length+name.length+data.length;
  }
  const directory=Buffer.concat(central),end=Buffer.alloc(22);end.writeUInt32LE(0x06054b50);end.writeUInt16LE(central.length/2,8);end.writeUInt16LE(central.length/2,10);end.writeUInt32LE(directory.length,12);end.writeUInt32LE(offset,16);
  return Buffer.concat([...local,directory,end]);
}
const base={
  'meta.json':JSON.stringify({version:'0.9',duration_s:10,event_count:2,frame_quality:{checked:4,suspect_ratio:.25}}),
  'samples.csv':'t_ms,lat,lon,speed_ms,vz\n0,31.23,121.47,4,0\n1000,31.2301,121.4701,4,1\n2000,31.2302,121.4702,4,-1\n3000,31.2303,121.4703,4,0',
  'events.csv':'id,t_ms,duration_ms,peak,lat,lon,speed_kmh,auto_label,final_label,frames,low_speed,segment_id,counts_toward_rhi\n0,1000,200,3,31.2301,121.4701,14.4,坑洼,坑洼,2,0,0,1\n1,2000,120,2,31.2302,121.4702,6,未定,未定,0,1,0,0',
  'segments.csv':'segment_id,kind,t_start_ms,t_end_ms,length_m,event_count,counts_toward_rhi\n0,局部密集段,1000,2000,4,2,1',
  'frames/ev0_t600.jpg':Buffer.from('deliberately invalid image for error UI'),
  'frames/ev0_t800.jpg':Buffer.from('deliberately invalid image for error UI'),
};
const fixture=zip(base),pack=api.parsePack(arrayBuffer(fixture),'viewer_fixture');
test('Quoted CSV, BOM, multiline and escaped quotes',()=>{
  const row=api.csvParse('\uFEFFid,note\r\n1,"a,b\n他说""好"""\r\n')[0];assert.equal(row.note,'a,b\n他说"好"');
  assert.throws(()=>api.csvParse('a,b\n"oops'),/引号/);
  assert.throws(()=>api.csvParse('a,b\n1,2,3'),/列数/);
});
test('STORE archive, actual events, frame names and speed-integral distance',()=>{
  assert.equal(pack.events.length,2);assert.equal(pack.stats.distanceM,12);assert.equal(pack.stats.lowCount,1);assert.equal(pack.stats.frameUsable,.75);assert.equal(pack.frames['0'][0].t_ms,600);assert.equal(pack.events[1].counts_toward_rhi,'0');
});
test('v0.6-v0.9 missing optional fields remain unknown, never fake zeros',()=>{
  for(const version of ['0.6','0.7','0.8','0.9']){
    const p=api.normalizePack({version},[],[{id:0,t_ms:0,auto_label:'未定'}],[],{},'old');assert.equal(p.stats.frameUsable,null);assert.equal(p.stats.distanceM,null);assert.equal(p.stats.lowCount,null);assert.ok(p.warnings.length);
    const archived=api.parsePack(arrayBuffer(zip({'meta.json':JSON.stringify({version}),'samples.csv':'t_ms,lat,lon,speed_ms\n0,,,','events.csv':'id,t_ms,auto_label\n0,0,未定'})),'old');assert.equal(archived.meta.version,version);assert.equal(archived.stats.density,null);
  }
});
test('Missing required files, compressed archive and truncation are rejected',()=>{
  assert.throws(()=>api.parsePack(arrayBuffer(zip({'meta.json':'{}'})),'x'),/缺少/);
  assert.throws(()=>api.zipIndex(arrayBuffer(zip(base,8))),/STORE/);
  assert.throws(()=>api.zipIndex(arrayBuffer(fixture.subarray(0,fixture.length-5))),/不完整/);
  assert.throws(()=>api.zipIndex(arrayBuffer(zip({'../x':'bad'}))),/路径/);
});
test('Invalid and duplicated event IDs are rejected',()=>{
  assert.throws(()=>api.normalizePack({},[],[{id:0,t_ms:0},{id:0,t_ms:1}],[],{},'x'),/重复/);
  assert.throws(()=>api.normalizePack({},[],[{id:0,t_ms:''}],[],{},'x'),/有效/);
  assert.throws(()=>api.normalizePack({},[],[{id:0.5,t_ms:0}],[],{},'x'),/有效/);
});
const vision={schema:'roadcheck.vision_review.v0',generated:'human_example',reviews:[{pack_id:'viewer_fixture',event_id:0,visual_label:'井盖',confidence:'high',why:'用于校验的人工示例',agree_with_imu:false}]};
const report={schema:'roadcheck.segment_report.v0',generated:'human_example',reports:[{pack_id:'viewer_fixture',segment_id:0,headline:'测试报告',score_note:'待定',why:'测试用例',comparison:'无对比数据',maintenance:[{priority:1,text:'复核',basis:'测试输入'}]}]};
test('Both example and real LLM JSON use the same renderer contract',()=>{
  api.validateResult(vision,'vision',pack);api.validateResult({...vision,generated:'llm'},'vision',pack);api.validateResult(report,'report',pack);
  const schema=JSON.parse(fs.readFileSync(path.join(__dirname,'../viewer/result.schema.json'),'utf8'));
  assert.equal(schema.$defs.source.enum.join(','),'llm,human_example');
  const labels=fs.readFileSync(path.join(__dirname,'../labels/schema.md'),'utf8');
  for(const label of schema.$defs.review.properties.visual_label.enum)assert.ok(labels.includes(label));
});
test('Cross-pack, unknown IDs, unknown labels, missing source and malformed reports rejected',()=>{
  const clone=o=>JSON.parse(JSON.stringify(o));let x=clone(vision);x.reviews[0].pack_id='wrong';assert.throws(()=>api.validateResult(x,'vision',pack),/pack_id/);
  x=clone(vision);x.reviews[0].event_id=99;assert.throws(()=>api.validateResult(x,'vision',pack),/不存在/);
  x=clone(vision);x.reviews[0].visual_label='水泥板错台';assert.throws(()=>api.validateResult(x,'vision',pack),/词表/);
  x=clone(vision);delete x.generated;assert.throws(()=>api.validateResult(x,'vision',pack),/generated/);
  x=clone(report);x.reports[0].maintenance[0].priority=-1;assert.throws(()=>api.validateResult(x,'report',pack),/priority/);
});
test('Blank GPS is not silently interpreted as zero',()=>{
  const p=api.normalizePack({},[{t_ms:0,lat:'',lon:''}],[],[],{},'x');assert.equal(p.track.length,0);
});
// Large synthetic archive: 3146 frames and 60 Hz samples, ~72 MB. No real imagery or coordinates.
const large={...base};let sampleText='t_ms,lat,lon,speed_ms,vz\n';for(let i=0;i<120000;i++)sampleText+=`${(i*1000/60).toFixed(1)},31.23,121.47,4,0.1\n`;large['samples.csv']=sampleText;
for(let i=0;i<3146;i++)large[`frames/sweep_t${i*2000}.jpg`]=Buffer.alloc(22000,100);
const big=zip(large),start=performance.now();api.parsePack(arrayBuffer(big),'large_fixture');const elapsed=performance.now()-start;
test('Large synthetic pack parses under 10 seconds',()=>assert.ok(elapsed<10000));
const dir=path.join(__dirname,'../userdata/viewer-test');fs.mkdirSync(dir,{recursive:true});
fs.writeFileSync(path.join(dir,'viewer_fixture.zip'),fixture);fs.writeFileSync(path.join(dir,'compressed.zip'),zip(base,8));fs.writeFileSync(path.join(dir,'large_fixture.zip'),big);
fs.writeFileSync(path.join(dir,'event_vision.json'),JSON.stringify(vision));fs.writeFileSync(path.join(dir,'segment_report.json'),JSON.stringify(report));
console.log(`${passed} groups passed; ${(big.length/1024/1024).toFixed(1)} MiB synthetic pack parsed in ${elapsed.toFixed(0)} ms. Browser fixtures: userdata/viewer-test/`);
