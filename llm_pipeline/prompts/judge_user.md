# L2 判读 · 用户提示模板

每事件一条，`{imu_block}` 由 L0 预计算注入（见 imu_features.build_feature_block），
三帧图片以 OpenAI 兼容格式的 image_url 附在本文本之后。

## 模板正文

```
事件 #{event_id}。以下为车轮上方前视摄像头在同一事件的 3 帧画面
（第 1 帧：冲击前 {a_off:+d}ms；第 2 帧：冲击即时 {b_off:+d}ms；第 3 帧：冲击后最近帧 {c_off:+d}ms，
负值表示该帧摄于冲击发生之前，缺陷本体通常位于画面中下部前方）。

{imu_block}

请按系统提示的词表与规则输出 JSON 判读。
```

## 注入方式（judge.py 实现）

- 文本部分作为 user content 的 text 项
- 三帧按 pre / impact / post_proxy 顺序作为 image_url 项（data:image/jpeg;base64,…）
- event_id 只出现在文本中（帧文件名不喂给模型，避免它用文件名作弊）
