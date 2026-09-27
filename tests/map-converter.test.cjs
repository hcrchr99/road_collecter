const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(require('node:path').join(__dirname,'../viewer/index.html'),'utf8');
const ctx=vm.createContext({});
vm.runInContext(html.match(/<script id="core">([\s\S]*?)<\/script>/)[1]+'\nthis.create=createMapConverter;',ctx);
const points=Array.from({length:121},(_,i)=>[120+i/10000,30]);
const ok=chunk=>({status:'complete',result:{locations:chunk}});
const limited={status:'error',result:{info:'CUQPS_HAS_EXCEEDED_THE_LIMIT',infocode:'10021'}};
async function main(){
  let time=0,starts=[],batches=[];
  const converter=ctx.create({now:()=>time,sleep:async ms=>{time+=ms;},request:async chunk=>{starts.push(time);batches.push(chunk.length);return ok(chunk);}});
  const output=await converter.convert([...points,points[0]]);
  assert.equal(output.length,122);assert.deepEqual(batches,[40,40,40,1]);
  assert.ok(starts.every((t,i)=>!i||t-starts[i-1]>=350));
  assert.deepEqual(JSON.parse(JSON.stringify(output[121])),points[0]);
  await converter.convert(points);assert.equal(starts.length,4);
  converter.clear();await converter.convert([points[0]]);assert.equal(starts.length,5);
  console.log('PASS 3 QPS pacing, batch size, deduplication, order, cache and clear');

  let calls=0;const waits=[],messages=[];
  const retry=ctx.create({now:()=>time,sleep:async ms=>{waits.push(ms);time+=ms;},request:async chunk=>++calls<=2?limited:ok(chunk)});
  await retry.convert([points[0]],()=>true,s=>messages.push(s));
  assert.equal(calls,3);assert.deepEqual(waits,[1000,2000]);assert.ok(messages.some(s=>s.includes('自动重试')));
  calls=0;const exhausted=ctx.create({now:()=>time,sleep:async ms=>{time+=ms;},request:async()=>{calls++;return limited;}});
  await assert.rejects(exhausted.convert([points[0]]),/已自动重试 4 次/);assert.equal(calls,5);
  console.log('PASS exponential backoff, retry status and bounded retries');

  calls=0;let fail=true;const sizes=[];
  const partial=ctx.create({now:()=>time,sleep:async ms=>{time+=ms;},request:async chunk=>{calls++;sizes.push(chunk.length);return fail&&calls===2?{status:'error',result:'INVALID_USER_KEY'}:ok(chunk);}});
  await assert.rejects(partial.convert(points.slice(0,41)),/INVALID_USER_KEY/);
  assert.equal(calls,2);fail=false;await partial.convert(points.slice(0,41));assert.deepEqual(sizes,[40,1,1]);
  console.log('PASS non-limit errors not retried, successful batches survive failure');

  let active=true;calls=0;
  const cancel=ctx.create({now:()=>time,sleep:async ms=>{time+=ms;active=false;},request:async()=>{calls++;return limited;}});
  await assert.rejects(cancel.convert(points,()=>active),/已取消/);assert.equal(calls,1);
  const serialStarts=[];
  const serial=ctx.create({now:()=>time,sleep:async ms=>{time+=ms;},request:async chunk=>{serialStarts.push(time);return ok(chunk);}});
  await Promise.all([serial.convert([points[0]]),serial.convert([points[1]])]);
  assert.ok(serialStarts[1]-serialStarts[0]>=350);
  console.log('PASS cancellation during retry and shared queue between connections');

  const plain=value=>JSON.parse(JSON.stringify(value));
  for(const mutation of ['replace','in-place']){
    let attempts=0;const source=points.slice(0,41).map(p=>p.slice()),original=plain(source);
    const mutating=ctx.create({now:()=>time,sleep:async ms=>{time+=ms;},request:async chunk=>{
      attempts++;
      assert.deepEqual(plain(chunk),attempts<=2?original.slice(0,40):original.slice(40));
      const locations=chunk.map(p=>({getLng:()=>p[0]+.001,getLat:()=>p[1]+.001}));
      // Capture numeric values before mutating request pairs.
      const result=locations.map(p=>({lng:p.getLng(),lat:p.getLat()}));
      if(mutation==='replace')chunk.splice(0,chunk.length,...result);
      else chunk.forEach(p=>{p[0]+=1;p[1]+=1;});
      return attempts===1?limited:{status:'complete',result:{locations:result}};
    }});
    const converted=await mutating.convert(source);
    assert.deepEqual(plain(source),original);
    assert.deepEqual(plain(converted),original.map(p=>[p[0]+.001,p[1]+.001]));
    assert.equal(attempts,3);
    converted[0][0]=0;
    const cached=await mutating.convert(source);
    assert.equal(attempts,3);assert.equal(cached[0][0],original[0][0]+.001);
  }
  console.log('PASS SDK array replacement and pair mutation, retry isolation, stable cache keys and copied output');
}
main().catch(e=>{console.error(e);process.exitCode=1;});
