# 部署说明（Vercel）

采集端的部署目标是**一个零构建的静态站点**：Vercel 直接把仓库根目录当输出目录发布，
没有 `npm install`、没有构建步骤、没有 Node 依赖。

## 站点结构

```
/                      ← index.html（采集端，站点入口）
/docs/项目启动方案.html   ← 赛事要求、排期、分工
/docs/视觉融合方案_v0.2.html
/manifest.webmanifest  ← PWA 清单（可加桌面图标）
/favicon.svg 等         ← 图标
```

Python 分析流水线（`*.py`）不参与部署，已在 `.vercelignore` 中排除。
这意味着线上站点只暴露采集端与两份文档，源码不会被公开下载。

## 为什么根路径必须是 index.html

Vercel 的静态站点只在访问路径上找 `index.html`。若文件名是 `collector.html`，
访问 `https://<项目>.vercel.app/` 会得到 404，必须手输 `/collector.html` 才能打开——
而采集端是要在骑行途中用手机打开的，多输一段路径不现实。

因此仓库把 `collector.html` 改名为 `index.html` 放在根目录。
`vercel.json` 里保留了一条 301 重定向，旧链接 `/collector.html` 仍可用。

## 首次部署

### 方式一：接入 Git（推荐，推送到 main 自动发布）

1. 打开 <https://vercel.com/new>，用 GitHub 账号登录
2. 选择本仓库 `hcrchr99/road_collecter`，点 Import
3. **Framework Preset 选 `Other`**，Build Command、Output Directory、Install Command 全部留空
   （`vercel.json` 已声明，通常不需要手动填）
4. 点 Deploy，约 10 秒后得到 `https://<项目名>.vercel.app`

之后每次推送到 `main` 都会自动重新部署。

### 方式二：命令行

```bash
npm i -g vercel
vercel login
vercel --prod
```

在项目根目录执行，交互提示里 Build Command 与 Output Directory 均直接回车跳过。

## 关键配置说明

`vercel.json` 中有三处不能删：

| 配置 | 作用 |
|---|---|
| `"outputDirectory": "."` | 把仓库根目录整体作为静态产物发布 |
| `"cleanUrls": false` | **必须关掉**。否则 Vercel 会把 `/docs/项目启动方案.html` 重写成无扩展名的路径，中文文件名下容易 404 |
| `"redirects"` | 把旧的 `/collector.html` 永久重定向到根路径，保证已分享的二维码/链接不失效 |

另外 `headers` 里显式放开了 `accelerometer` / `gyroscope` / `geolocation` / `camera`
四项 `Permissions-Policy`。浏览器默认是 `self`，这里写出来是为了防止将来有人加站点级
策略时把传感器权限一起关掉——采集端**没有这几项权限就完全无法工作**。

> 注意：`Permissions-Policy` 管的是"同源页面能否调用"，**不能**替代 HTTPS。
> 传感器与定位 API 要求安全上下文，`http://` 或 `file://` 下浏览器会直接拒绝授权。

## 部署后自检清单

在**手机**上打开站点，确认：

- [ ] 地址栏是 `https://`（不是 http）
- [ ] 页面显示的 API 自检项全部为可用
- [ ] 点授权后，定位 / 摄像头 / 运动传感器三项系统权限都弹了窗
- [ ] 试采 4 秒能报出实测采样率（正常在 50~100 Hz）
- [ ] 若停留在试采页，去掉 `--max-time` 后长时间骑行仍持续记录

## 常见问题

**打开是一串文件列表或 404**
`index.html` 不在根目录，或 `outputDirectory` 被改成了别的目录。

**手机上提示"需要 HTTPS"**
Vercel 默认给 `*.vercel.app` 签证书。若绑定了自定义域名，需在域名服务商处加 CNAME
并等 Vercel 自动签发证书生效。

**文档页面 404**
检查 `cleanUrls` 是否为 `false`。中文文件名经 URL 重写后极易失效。

**传感器没有数据**
八成是权限问题，不是部署问题。先在手机上打开站点，逐项完成"环境自检"。
iOS 还要求传感器授权必须由用户手势触发——不能自动申请。
