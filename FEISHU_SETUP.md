# 飞书群聊接入

## 1. 飞书开放平台

1. 创建企业自建应用并开启机器人能力。
2. 为应用添加权限：
   - `im:message.group_at_msg:readonly`
   - `im:message:send_as_bot`
   - `im:message:update`（持续更新同一张进度卡片）
   - `wiki:wiki`
   - `docx:document`
   - `tenant:tenant:readonly`
   - `tenant:tenant.domain:read`
3. 在事件订阅中选择“使用长连接接收事件”。
4. 订阅 `im.message.receive_v1`。
5. 发布应用，并把机器人加入测试群。

## 2. 后端配置

复制 `.env.example` 中需要的变量到启动环境。当前代码直接读取环境变量，不会自动读取 `.env` 文件。

PowerShell 示例：

```powershell
$env:FEISHU_APP_ID="cli_xxx"
$env:FEISHU_APP_SECRET="xxx"
$env:FEISHU_CONNECTION_MODE="long_connection"
```

### 明确的文档任务写入 Wiki

普通长回答仍会留在飞书聊天中；只有用户明确说“写成文档”“整理到 Wiki”或“输出完整报告”时，Agent 才会获得 Wiki 写入能力。

飞书的“创建知识空间”接口不支持机器人使用的应用身份令牌，因此需要先做一次人工准备：

1. 在飞书中创建项目知识空间。
2. 在知识空间设置中把机器人应用添加为管理员。
3. 从知识空间设置页地址复制知识空间 ID，例如
   `https://example.feishu.cn/wiki/settings/123456789` 中的 `123456789`。
   普通文档链接（例如 `https://my.feishu.cn/wiki/UNz...`）末尾是文档节点 token，不是知识空间 ID，不能填入 `FEISHU_WIKI_SPACE_ID`。
4. 配置知识空间 ID 和浏览器访问地址：

```powershell
$env:FEISHU_WIKI_SPACE_ID="123456789"
$env:FEISHU_WIKI_BASE_URL="https://example.feishu.cn/wiki" # 可选
```

如果每个群使用不同知识空间，可不设置全局 `FEISHU_WIKI_SPACE_ID`，在下面的群绑定请求中传 `wiki_space_id`。当应用只能访问一个知识空间时，后端也会自动识别该空间。开通企业域名权限后，后端会自动生成可点击的 Wiki 链接；无法开通时再手动设置 `FEISHU_WIKI_BASE_URL`。

如果需要把飞书群绑定到真实项目目录，先配置 `CT_ALLOWED_WORKSPACE_ROOTS`，启动后调用：

```http
POST /api/integrations/feishu/bindings
Content-Type: application/json

{
  "tenant_key": "飞书事件中的 tenant_key",
  "chat_id": "飞书事件中的 chat_id",
  "workspace_dir": "D:\\Project\\YourProject",
  "channel_name": "YourProject",
  "wiki_space_id": "123456789"
}
```

没有预先绑定的群会自动使用 `backend/data/workspaces/<channel_id>/` 隔离目录。

## 3. 启动与检查

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

检查：

- `GET /health`
- `GET /api/integrations/feishu/status`
- 在测试群发送 `@CatTogether 帮我规划这个项目`

机器人会先回复 Run ID，由入口 Agent 判断是否邀请其他角色协作；同一张卡片持续显示最新进展，工作过程默认折叠，完成后显示回复角色、最终答案和可选文档链接。

## 4. 运行测试

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

飞书凭证未配置时，长连接不会启动，但后端、原 REST/WebSocket 和自动化测试仍可使用。
