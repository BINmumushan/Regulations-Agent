# regulations-agent 公网部署指南（Render 免费版）

> 目标：获得一个固定公网链接（`https://xxx.onrender.com`），供他人试用半个月到一个月，完全免费。

---

## 第一步：推送代码到 GitHub（约 5 分钟）

本地 Git 仓库已初始化并提交完毕（含向量库数据），现在只需要发布到 GitHub：

1. 打开 **GitHub Desktop**（你电脑上已安装）
2. 点击菜单 **File → Add local repository**
3. 选择路径 `C:\Users\29526\Desktop\regulations-agent`
4. 点击 **Publish repository**：
   - 名称：`regulations-agent`（或任意）
   - **务必取消勾选 "Keep this code private"？不需要** —— 保持 **Private（私有）** 即可，Render 支持私有仓库
   - 点击 Publish

完成后代码就到了 `https://github.com/BINmumushan/regulations-agent`。

---

## 第二步：在 Render 上部署（约 10 分钟）

1. 打开 https://dashboard.render.com 注册/登录（可用 GitHub 账号一键登录）
2. 点击 **New → Web Service**
3. 选择 **Build and deploy from a Git repository**，连接你的 GitHub 账号，选中 `regulations-agent` 仓库
4. 配置页面：
   - **Name**：`regulations-agent`（决定域名，如 `regulations-agent.onrender.com`）
   - **Region**：Singapore（离国内最近）
   - **Runtime**：**Docker**（项目已带 Dockerfile，Render 会自动识别）
   - **Instance Type**：**Free**
5. 添加环境变量（在 Environment variables 部分）：

   | Key | Value |
   |-----|-------|
   | `OPENAI_API_KEY` | 你的 DeepSeek API Key（sk-开头） |
   | `OPENAI_BASE_URL` | `https://api.deepseek.com/v1` |
   | `LLM_MODEL` | `deepseek-v4-flash` |
   | `EMBEDDING_PROVIDER` | `local` |
   | `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` |
   | `DEMO_PASSWORD` | `111222`（或你想要的试用密码） |

6. 点击 **Create Web Service**，等待构建（首次约 5-10 分钟，需要下载依赖和模型）

---

## 第三步：验证（1 分钟）

构建完成后，访问 `https://regulations-agent.onrender.com`：

- 应该能看到问答界面
- 输入演示密码进入
- 提问测试回答是否正常

把这个链接发给别人即可！

---

## 免费版注意事项

| 事项 | 说明 |
|------|------|
| 冷启动 | 15 分钟无人访问会休眠，下次访问约 30-50 秒唤醒 |
| 额度 | 每月 750 小时免费，半个月完全够用 |
| 内存 | 512MB，当前应用可以跑，但上传大文件入库可能受限 |
| 数据 | 免费版无持久磁盘——别人通过 `/upload` 上传的文件重启后会丢，但不影响问答功能 |
| 更新 | 改了代码 push 到 GitHub，Render 自动重新部署 |

## 已部署后想关闭

Render Dashboard → 选择服务 → Settings → **Suspend**（暂停）或删除服务即可。

---

## 常见问题

**Q: 构建失败怎么办？**
查看 Render 的构建日志（Deploy 页面往下滚动），最常见的是依赖安装超时，重试一次即可。

**Q: 首次访问特别慢？**
首次请求需要加载 fastembed 模型（已在构建时预下载），第二个请求起就正常了。

**Q: 想换密码？**
Render Dashboard → Environment → 修改 `DEMO_PASSWORD` → 保存即自动重启生效。

**Q: 为什么不用原始 PDF？**
问答只需要 FAISS 向量库（已包含在仓库中）。原始 PDF（277MB）只在上传新文档重新入库时需要，在本地操作后重新 push 向量库即可。
