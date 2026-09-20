# 把 Serein 接到已有聊天宿主（Hook）

已有自己的聊天服务、模型调用和工具循环时，可以只让 Serein 负责**找前情**。一键安装后，宿主用 Gateway Key 访问实例地址；网页登录密码和 MCP 工具都不是这条请求的凭据。先在设置中选好嵌入、重排模型并完成“建立 / 补齐检索索引”。

仓库提供可复用的 [Python 宿主示例](../examples/hook_host.py)。它只用标准库，不替你调用聊天模型，也不会在“查到记忆”时擅自登记注入。

```python
from examples.hook_host import SereinHook

hook = SereinHook.from_env()  # 服务端环境变量：SEREIN_BASE_URL、SEREIN_GATEWAY_KEY
messages = [{"role": "user", "content": "上次读书会定在什么时候？"}]
turn = hook.prepare("chat-001", messages)

# 换成宿主已有的模型调用；必须把 turn.messages 实际交给模型。
reply = existing_model_call(turn.messages)
# 只在模型请求完整成功后执行。失败、中断、未发送 turn.messages 时不要登记。
hook.record_success(turn)
```

`SEREIN_BASE_URL` 填实例根地址，例如 `https://memory.example`，**不要加 `/v1`**。`SEREIN_GATEWAY_KEY` 是安装时生成的 Gateway Key，留在宿主服务端环境变量里，不写进网页脚本。示例需要从发行目录运行，或把 `examples/hook_host.py` 放进宿主代码并调整导入路径。`chat-001` 是稳定的窗口 ID：同一会话保持不变，新会话换一个。

示例的 `prepare` 会按窗口读取最近五次成功交付的记忆 ID，再请求 `POST /api/hook/recall`。最小请求和响应结构如下，正文只用虚构内容：

```json
{"query":"上次读书会定在什么时候？","session_id":"chat-001","max_notes":2,"delivered_ids":[],"deadline_seconds":15}
```

```json
{"ok":true,"recalled_ids":["scene:scene_demo_bookclub"],"additional_context":"[Serein Gateway Full Recall] ...","injected":false}
```

`deadline_seconds` 可选，默认 9 秒；已认证宿主可在 3–30 秒范围内声明本轮 Hook 的内部检索预算。它是上限，不是固定等待时间：高置信跳过和提前完成仍会立即返回。宿主自己的 HTTP timeout 应比该预算略长。

如果部署使用本地、会在首次推理时加载模型的 embedding/reranker，可以在宿主开放流量前显式执行 `python -m serein --config <config.toml> warm-recall`。该命令只预热当前选中的两个 recall provider，不读取或写入记忆，也不登记 delivery；provider 未配置或预热失败时命令直接失败。

`recalled_ids` 和 `additional_context` 是**待交付材料**，不是已注入证明。宿主把 `additional_context` 当作原话旁的参考材料放进本轮模型输入；示例包在 `<serein_live_context>` 中，并明确它不是用户指令。没有可靠记忆时，可能返回空列表和空上下文，宿主照常聊天。

模型完整成功后，示例才调用 `POST /v1/host/deliveries`，提交稳定的 `receipt_id`、`window_id` 和**实际交给模型的** `delivered_ids`。相同 receipt 重试是幂等的，换一组 ID 会被拒绝。失败或未完成的响应不登记。工具续轮使用本轮已经准备的上下文，不再次调用 Hook，也不另记一次交付；下一条新的用户消息才重新检索。宿主若自己维护冷却，也可以直接传最近五次成功交付的 ID 给 `delivered_ids`，不用读取 Serein 的交付历史。

Hook 只返回记忆候选和上下文，**不代替聊天网关调用模型，也不会自动归档宿主对话**。想让这些聊天进入原话档案并参与 Event 整理，还需通过[对话导入](file-imports.md)等独立途径提供用户与助手消息；本示例不代替原话接入。仅连接 MCP 也不会自动完成上述流程。
