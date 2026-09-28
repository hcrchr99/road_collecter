const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const html=fs.readFileSync(path.join(__dirname,'../viewer/index.html'),'utf8');
const section=html.match(/<section class="agent-intro"[\s\S]*?<\/section>/)?.[0];
assert.ok(section);assert.ok(html.indexOf(section)<html.indexOf('<div class="hero">'));
assert.equal((section.match(/<li>/g)||[]).length,5);
for(const text of ['读取事件图片和传感器特征。','调用模型判断路面类型，如坑洼、井盖、标线。','用规则检查结果，必要时补充图片再判读。','输出最终分类、置信度和理由，将部分事件交给人工复核。','保存判读结果和运行日志。','不代表正在执行','href="#agent"','href="#vision"'])assert.ok(section.includes(text),text);
console.log('PASS top Agent introduction: position, five steps, replay disclosure and navigation');
