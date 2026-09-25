# RoadCheck 道路观察台 v0.1

入口：`viewer/index.html`。单文件 HTML + 内联 CSS/JS，无 npm 依赖、无构建、无外部 CDN（主动开启的高德底图除外）。

## 启动与演示

在仓库目录执行 `python -m http.server 8765 --bind 127.0.0.1`，打开 `http://127.0.0.1:8765/viewer/`。
页面默认加载合成演示。直接双击 HTML 也可看演示；若浏览器禁止 file:// 下的 Worker，请用上述本地服务导入 ZIP。

演示顺序：

1. 查看四张概览卡，确认显眼的「合成演示」标签。
2. 点击轨迹上的事件 #0，照片默认在冲击前 200 ms。左右切帧，可回到冲击前 400 ms。
3. 点击事件 #11，看到过暗提示；筛选「仅低速记录」，仍能看到事件 #3。
4. 看第三屏的三条示例判读，点击「回看」跳回对应事件。
5. 切换第四屏路段，查看解释、建议与「没有第二趟数据」说明。
6. 展示第五屏的能力边界。所有 AI 卡片是明确标注的人工示例。

合成画面是程序绘制的示意图，不能代替真实帧、人工确认或模型预测。

## 数据输入

- 必需：原始 STORE ZIP（method=0）中的 `meta.json`、`samples.csv`、`events.csv`。
- 可选：`segments.csv`、`sweeps.csv`、`frames/*.jpg`。本版展示事件帧，不提供独立扫描帧标注功能。
- 帧名按 `frames/ev{id}_t{ms}.jpg` 匹配，events.frames 不是文件名。
- 包 ID = ZIP 文件名去掉 `.zip`；导入 AI JSON 时须精确匹配。重命名 ZIP 时要同步 JSON 的 pack_id。
- `auto_label` 保持算法原文；不把 `final_label` 当成人工真值（尤其 v0.9）。
- 缺失字段显示 N/A；低速记录不删除；`counts_toward_rhi=0` 显示为不计入 RHI。
- 大包在 Worker 解析，照片按需创建 Blob URL，用完释放，不批量解码整包图片。
- 原始包与 JSON 只留在页面内存，无后端上传、无 localStorage 持久化。

### 统计口径

里程优先读 `meta.distance_m`，否则对 `samples.speed_ms` 按时间做梯形积分，跳过无效速度及超过 5 秒的间隔，并报告覆盖不足。不是用 GPS 跳点直接累加里程。

每公里事件密度 = events.csv 实际行数 / 里程 km，包含低速记录；不等于确认病害密度。

画面可用率 = `1 - meta.frame_quality.suspect_ratio`（采集端抽查口径）。它不一定等于交接材料中的人工复核「87.9%」；真实包到位后需确认统计分母及来源，不能为了匹配数字硬编码。

## AI 结果契约

通过第三屏、第四屏分别导入 JSON。校验 schema、generated、字段类型、真值词表、唯一 ID、包 ID 与被引用事件/路段。校验失败保留原状态，显示错误。

机器可读契约见 `result.schema.json`（JSON Schema 2020-12）；页面内置同等字段检查，并额外核对当前包的引用关系。

`generated` 仅接受 `llm` 或 `human_example`；来源是文件提供方声明，不是签名验证。真实 LLM 使用相同渲染层，不需要改 UI。

视觉示例（请换成当前包中的真实 ID 和经人工确认的内容）：

```json
{
  "schema": "roadcheck.vision_review.v0",
  "generated": "human_example",
  "reviews": [{
    "event_id": 0,
    "pack_id": "你的采集包文件名（不含.zip）",
    "visual_label": "遮挡/无法判定",
    "confidence": "low",
    "why": "示例占位，待人工回看真实画面后填写。",
    "agree_with_imu": false
  }]
}
```

报告示例：

```json
{
  "schema": "roadcheck.segment_report.v0",
  "generated": "human_example",
  "reports": [{
    "segment_id": 0,
    "pack_id": "你的采集包文件名（不含.zip）",
    "headline": "示例占位，待核对路段数据。",
    "score_note": "相对严重度：待分析端提供",
    "why": "待填写可追溯依据。",
    "maintenance": [{"priority": 1, "text": "先复核现场证据", "basis": "当前证据尚不足"}],
    "comparison": "尚无多趟对齐结果。"
  }]
}
```

视觉标签必须使用 `labels/schema.md` v1.1 的十类词表；例如「设计接缝」合法，「水泥板错台」不能作为新标签，但可在 why 中说明。分数由分析端负责，展示端不生成官方分数。

## 地图

默认是无底图的离线相对轨迹图（不是可验收的在线合规底图）。可选展开「连接高德地图」：填 Web JS API Key，加安全代理 URL（推荐）或开发用 securityJsCode，再明确允许坐标交给高德。

浏览器原始坐标选 WGS-84；服务分批调用 `AMap.convertFrom` 后以 GCJ-02 上图。已转为 GCJ-02 的包手动选择 GCJ-02，避免重复转换。在线底图显示轨迹、事件和路段；失败保留离线图。更换采集包后需重新连接。SDK 加载后如需更换 Key，请刷新页面。

配置不入代码、不保存到磁盘。正式部署应配置安全代理与域名白名单。
参考：[高德安全密钥](https://lbs.amap.com/api/javascript-api-v2/guide/abc/jscode)、[基础类及坐标系](https://lbs.amap.com/api/javascript-api-v2/guide/abc/basetype)。

## 验证和剩余验收

运行 `node tests/viewer.test.cjs`：覆盖 CSV 转义、STORE ZIP、旧包缺字段、压缩包/截断拒绝、ID/词表/跨包校验及约 73 MB 合成包解析。测试文件仅生成于 `userdata/viewer-test/`，已由既有 Git/Vercel 规则排除。该大包的帧为占位字节，仅测解析吞吐，不测真实照片质量。

还需用团队指定的 71.6 MB v0.8 包、v0.9 实包回归，并与分析端核对里程和质检口径；需要真实地图配置完成在线 API 联调。当前不能据合成测试声称实包验收通过。

本版未实现账号、多人协作、真实 LLM 调用或官方评分；多趟对比仅展示导入报告中的 comparison，不在前端计算跨包匹配。

## 部署

沿用现有 Vercel 静态配置，公开入口为 `/viewer/index.html`。Git 推送 main 可能触发部署，因此在团队确认前不自动提交或推送。真实 ZIP/标注文件不要加入 Git，演示公开前先脱敏。首次发布后核对 `/viewer/` 与采集端入口的跳转。
