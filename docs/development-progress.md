# 开发进度

## 当前开发阶段

当前处于“统一 `AgentConfig`、基础 Provider/Tool 构件、最小非流式 AgentRunner、最小持久化会话、队列式消息循环与本地 HTTP/CLI”阶段。根目录 `.env` 被加载为唯一的 `AgentConfig`，其中包含 Provider、本地工作目录和 HTTP API 子配置。当前 Provider 层同时包含 OpenAI-compatible 适配器和原生 Ollama 适配器；默认测试不连接真实 Provider 服务、Ollama、GPU 或视觉模型。

## 已完成功能

### 2026-09-10

- 以 `PROVIDER_TYPE`、`PROVIDER_API_KEY`、`PROVIDER_API_BASE`、`PROVIDER_MODEL`、`PROVIDER_MAX_TOKENS` 和 `PROVIDER_TEMPERATURE` 替代旧的特定服务配置；`.env` 保持本地忽略，`.env.example` 提供无密钥示例。
- 新增 `load_provider_config()`：仅在调用时读取根目录 `.env`，进程环境变量优先，并交由 `ProviderConfig`/Pydantic 校验类型、Provider 类型和必填项。
- 保留 ProviderConfig 的内部默认字段命名，同时通过 loader 映射项目级环境变量，不把配置绑定到某一个模型服务。
- 保留 `openai` SDK，并移除当前源码不再引用的 `anthropic`、`ollama` 和 `jsonschema` 直接依赖；`uv.lock` 已同步。
- 更新 setuptools 包列表，使当前 `config`、`providers`、`tools` 与 `tools.builtin` 包被包含。
- 将 Ruff 行长规则调整为 100 字符，解决当前 Provider 源码中的旧 E501 行长提示。
- 新增 5 个离线配置加载测试，覆盖统一变量映射、`.env` 加载、进程环境覆盖、空 API Key、不支持的 Provider 类型和缺失必填值；当前默认 pytest 测试均通过。
- 统一 `LLMProvider` 抽象接口与 OpenAI-compatible 适配器的方法命名为 `chat()` 和 `stream_chat()`，消除适配器无法满足抽象接口契约的问题。
- 补全 `src.tools.Tool` 的公共导出，使默认 `ProviderFactory` 可正常导入并创建 OpenAI-compatible 适配器；新增无网络的工厂构造测试，覆盖抽象接口契约和 OpenAI-compatible 的空 API Key 本地端点场景。
- 移除 Anthropic-compatible Provider、Anthropic Tool Schema 与 `anthropic` 依赖，当前仅保留 OpenAI-compatible Provider 支持。
- 新增 29 个 Tool 子系统离线测试，覆盖 Tool 定义与 OpenAI Function Schema、注册表参数校验与执行错误、取消传播，以及内建 Tool 发现和过滤。
- 新增最小非流式 `AgentRunner`：每轮固定调用 `provider.chat()`，顺序处理 Tool Call 与 Tool Result；不包含 `stream_chat`、目标模式、消息注入或 Tool Call 回调。新增 5 个离线 Runner 测试，覆盖普通响应、工具循环、禁用工具、最大迭代边界与非流式调用路径。
- 新增最小持久化会话：`Session` 仅保存消息历史，`JsonlSessionStorage` 和 `SessionManager` 不再维护 `summary`、`summary_until`、`goal_state` 或目标状态相关 API。旧 JSONL Header 中的这三个字段会在读取时忽略，并在下一次保存时移除；新增 4 个离线生命周期测试。

### 2026-09-11

- 新增项目级 `WORKSPACE_PATH` 本地配置；随后将 Provider、API 与工作目录收敛到根级 `AgentConfig`。`load_agent_config()` 一次读取 `.env`/进程环境变量并构造嵌套 `ProviderConfig`、`ApiConfig` 与 `workspace_path`；未传入路径的 `SessionManager()` 从该根级配置取得工作目录。仓库根目录 `workspace/` 已被 Git 忽略，默认测试覆盖路径加载、优先级、空值校验和 Manager 回退行为。
- 新增最小 `MessageBus`、`ContextBuilder` 和队列式 `AgentLoop`：主循环持续消费入站队列，并为每条消息创建受追踪、可取消的处理任务；不同会话可并行处理，同一 `session_id` 从读取历史到保存结果使用锁保持顺序。每条入站消息按“系统提示词 + 非系统历史 + 当前用户消息”构建请求，交由非流式 `AgentRunner` 执行，再保存本轮新增消息并发布出站回复。系统提示词不写入 Session；仅在 `AgentRunner` 正常返回 `AgentRunResult` 后保存本轮新增消息，Provider 或 Runner 异常只发布不泄露内部细节的失败回复。新增离线 Context 与 Loop 测试。
- `ApiConfig` 作为 `AgentConfig.api` 管理 `API_HOST`、`API_PORT` 和 `API_REQUEST_TIMEOUT_SECONDS`，默认仅监听 `127.0.0.1:8000`；`.env.example` 已同步安全示例。
- 重构本地 `HttpApiService`：`POST /v1/messages` 仅发布当前定义的 `InboundMessage(session_id, content, metadata)`，由唯一出站路由任务按私有关联标识等待对应 `OutboundMessage`；旧的 `channel`、`chat_id`、`sender_id` 和认证逻辑均已移除。新增会话查询与 `POST /v1/sessions/clear`；清空操作直接调用 `SessionManager.clear_messages()` 重置并持久化消息历史，不经过 `MessageBus` 或 AgentLoop。
- 新增最小 `Application`、`robot-agent` 控制台入口和 `python -m src` 入口。应用只使用 `AgentConfig` 装配当前已有的 Provider、SessionManager、ToolRegistry、MessageBus、AgentLoop 和 HTTP 服务；未恢复 Channel、Cron、MCP、Memory、Subagent 或复杂运行时。启动时会先确认 AgentLoop 未立即失败，再开放 HTTP 监听。
- `aiohttp` 已加入运行时依赖并同步 `uv.lock`。HTTP/CLI 离线测试覆盖请求关联、并发响应、超时、路由关闭、真实 AgentLoop 装配、会话 API、启动失败与 CLI 生命周期；Ruff、`compileall` 与 79 项默认测试均已通过，`python -m src --help` 和已安装的 `robot-agent --help` 已验证。
- 新增原生 `OllamaCompactProvider`，并通过 `PROVIDER_TYPE=ollama` 纳入统一 `AgentConfig` 和 Provider 工厂。该适配器使用官方 `ollama` SDK，沿用通用 `PROVIDER_API_BASE`、`PROVIDER_MODEL`、`PROVIDER_MAX_TOKENS`、`PROVIDER_TEMPERATURE` 与 `PROVIDER_REQUEST_TIMEOUT_SECONDS` 配置；本地 Ollama 通常不需要 `PROVIDER_API_KEY`。
- `ToolMessage` 增加可选的单张本地 `image_path`。图片二进制不进入核心消息或会话数据；仅原生 Ollama 适配器会在发送请求前读取该本地文件，并作为一张工具结果图片交给 Ollama SDK。OpenAI-compatible 适配器不支持此字段。离线测试使用 Mock/Fake Ollama Client 和临时图片文件，不运行真实 Ollama、网络、GPU 或设备测试。
- 本轮新增 Provider、配置、工具结果路径传递与 JSONL 兼容测试；当前全量默认 `pytest` 为 101 项通过，`compileall` 与 Ruff 检查、格式检查均通过。
- 新增无参数内建 `capture_camera` 工具。它通过 `ToolLoader` 在 `Application` 装配时自动注册，返回固定的本地图片路径 `./workspace/pictures/test.jpg`，以供 Tool → ToolMessage → 原生 Ollama 图片链路使用；当前不读取图片、不连接机器人头部摄像头，也不执行任何硬件控制。

## 待开发功能

- Provider 生命周期与真实服务端到端集成测试；
- 面向真实 Ollama 视觉模型的端到端联调；
- 更多具体内建工具及其权限、失败处理和测试；
- 流式 Agent 执行、目标模式、消息注入、运行时调度和任务编排；
- ROS 2、机器人控制和硬件适配；
- 外部系统连接和通信协议。

## 待优化项

- 在实际 Provider 联调需求明确后，补充超时、重试、观测和错误映射策略。
- 在 Provider 具体能力确认后，再决定是否需要 Provider 专属配置。
- 继续为 Provider、Tool 与 AgentRunner 补充独立、无网络的接口测试。
- 如需扩展多模态能力，先明确多张图片、远程资源和其他 Provider 的消息格式；当前工具结果仅支持单张本地图片路径，且仅原生 Ollama 适配器可发送该图片。
- 如未来引入第二个传输层或出站消费者，先为 `MessageBus` 设计按订阅者或路由分发的机制；当前 HTTP 服务是唯一的出站消费者。

## 待解决问题

- 本地 `.env` 中必须由用户填写与所选 Provider 匹配的 API Base、模型名，以及托管服务所需的 API Key。
- 当前默认测试验证配置加载、Provider 工厂构造、Tool 基础设施、非流式 AgentRunner、ContextBuilder、MessageBus/AgentLoop、HTTP/CLI 与 Session 持久化生命周期；不会验证凭据有效性、模型可用性或实际 Provider 网络连接。
- 本地 Ollama 服务地址、可用模型和视觉模型能力由使用者自行准备；默认测试不会验证其可访问性或实际图片理解结果。
