# robot-agent

## 当前状态

项目当前处于“统一 Agent 配置、基础 Provider/Tool 构件、最小非流式消息循环和本地 HTTP/CLI 入口”阶段。现有源码包括：

- 基于 Pydantic 的根级 `AgentConfig`，其中包含 `ProviderConfig`、`ApiConfig` 与工作目录；
- 从根目录 `.env` 和进程环境变量加载完整 `AgentConfig` 的函数；
- OpenAI-compatible Provider 与原生 Ollama Provider 适配器源码；
- 通用消息模型、工具抽象、工具注册表和内建工具发现基础设施。
- 最小持久化会话：仅维护消息历史的 `Session`、JSONL 存储和 `SessionManager`。
- 内存 `MessageBus`、`ContextBuilder` 和队列式 `AgentLoop`：每条消息均构建临时系统提示词、读取会话历史、执行 `AgentRunner`、持久化结果并发布出站回复。
- 本地 HTTP/CLI 装配源码：HTTP 请求经由 `MessageBus` 发布 `InboundMessage`，等待关联的 `OutboundMessage`；`Application` 负责装配 AgentLoop、会话、Provider、总线与 HTTP 服务。

当前包含最小、非流式的 `AgentRunner`：它按轮调用 `provider.chat()`，顺序处理模型请求的工具调用，并在得到最终响应或达到最大迭代次数时结束。

`Session` 不维护摘要、摘要边界或目标状态；旧 JSONL 会话文件中的这些已移除 Header 字段会在读取时忽略，并在下一次保存时移除。

`MessageBus` 的入站队列会被 `AgentLoop` 连续消费，每条消息由受追踪、可取消的后台任务处理；不同会话可并行处理，同一 `session_id` 从读取历史到保存结果始终串行化。每次请求只向 Provider 发送一个临时系统消息；它不会写入 Session。只要 `AgentRunner` 正常返回 `AgentRunResult`，会话历史、当前用户消息和本轮新增的 Assistant/Tool 消息就会按顺序持久化，即使最终回复为空或达到最大迭代边界。Provider 或 AgentRunner 抛出异常时，循环只发布不含内部细节的失败回复，不写入 Session。

HTTP 层为每个请求写入私有的关联标识，并由唯一的出站路由任务将 `OutboundMessage` 交回对应等待者，因此并发请求不会互相取走响应。清空会话直接调用 `SessionManager.clear_messages()` 重置并持久化消息历史；它不经过 `MessageBus` 或 AgentLoop。清空后保留会话标识，但持久化消息历史为空。

尚未实现流式 Agent 执行、目标模式、消息注入、运行时调度、更多具体内建工具、设备控制、ROS 2 或硬件适配。Provider（包括 Ollama）的真实服务联调也不会在默认测试中执行。

## 目录结构

```text
robot-agent/
├── AGENTS.md
├── README.md
├── .env.example
├── pyproject.toml
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── context.py
│   │   ├── loop.py
│   │   └── runner.py
│   ├── bus/
│   │   ├── __init__.py
│   │   ├── message_bus.py
│   │   └── messages.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── schema.py
│   ├── providers/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── messages.py
│   │   ├── ollama_compact_provider.py
│   │   ├── openai_compat_provider.py
│   ├── session/
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   ├── models.py
│   │   └── storage.py
│   └── tools/
│       ├── base.py
│       ├── context.py
│       ├── loader.py
│       ├── registry.py
│       └── builtin/
│           └── capture_camera.py
└── tests/
    ├── agent/
    │   ├── test_context.py
    │   ├── test_loop.py
    │   └── test_runner.py
    ├── config/
    │   └── test_loader.py
    ├── providers/
    │   ├── test_factory.py
    │   └── test_ollama_compact_provider.py
    ├── session/
    │   └── test_session_lifecycle.py
    └── tools/
        ├── test_base.py
        ├── test_tool_loader.py
        └── test_registry.py
```

`robot-agent` 是项目/发行名称，`src` 是 Python 导入包名。当前打包配置包含 `src.agent`、`src.bus`、`src.config`、`src.providers`、`src.session`、`src.tools` 和 `src.tools.builtin`。

本轮还增加了 `src/__main__.py`、`src/api/`（`service.py`）和 `src/cli/`（`application.py`、`main.py`），以及对应的 `tests/api/` 与 `tests/cli/` 测试目录；打包配置已包含 `src.api` 与 `src.cli`，并声明 `robot-agent` 控制台入口。

## Python 与 uv

项目支持 Python 3.10（含）至 3.13（不含），并通过 `uv` 管理依赖和虚拟环境。

```powershell
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

运行时依赖包括：

- `openai`：OpenAI-compatible Provider；
- `ollama`：原生 Ollama Provider；
- `aiohttp`：本地 HTTP API；
- `pydantic`：`AgentConfig` 及其子配置校验；
- `python-dotenv`：根目录 `.env` 加载。

Ruff 的项目行长规则为 100 个字符。`.env` 是本机配置文件，已被 Git 忽略；只能提交 `.env.example`，不能提交密钥或个人端点配置。

## 项目配置

根目录 `.env` 使用项目级变量构造唯一的 `AgentConfig`，其中包含 Provider、工作目录和本地 API 子配置，而非特定模型服务的配置对象：

```dotenv
WORKSPACE_PATH=./workspace
PROVIDER_TYPE=openai_compat
PROVIDER_API_KEY=
PROVIDER_API_BASE=https://api.openai.com/v1
PROVIDER_MODEL=your-model-name
PROVIDER_MAX_TOKENS=1024
PROVIDER_TEMPERATURE=0.7
PROVIDER_REQUEST_TIMEOUT_SECONDS=60
API_HOST=127.0.0.1
API_PORT=8000
API_REQUEST_TIMEOUT_SECONDS=60
```

当前支持的 `PROVIDER_TYPE` 为：

- `openai_compat`
- `ollama`：使用原生 Ollama SDK；此类型可将 `PROVIDER_API_KEY` 留空，并通常使用本地 `PROVIDER_API_BASE`，例如 `http://127.0.0.1:11434`。

调用 `src.config.load_agent_config()` 时才会读取 `.env`；已存在的进程环境变量优先于 `.env` 值。`PROVIDER_*` 环境变量会被映射为 `AgentConfig.provider`（`ProviderConfig`），`API_*` 会被映射为 `AgentConfig.api`（`ApiConfig`），`WORKSPACE_PATH` 会被映射为 `AgentConfig.workspace_path`。`WORKSPACE_PATH` 是必填的本地工作目录；未显式传入路径的 `SessionManager()` 会从 `AgentConfig` 使用它，且相对路径按当前工作目录解析。

对于不需要凭据的本地兼容端点，`PROVIDER_API_KEY` 可以留空；托管服务通常需要由本机用户填写真实 API Key。应用不得记录或输出该值。

使用原生 Ollama 时，仍使用同一组通用 `PROVIDER_*` 配置，而不是新增 Ollama 专属配置对象。例如可在本地 `.env` 中设置：

```dotenv
PROVIDER_TYPE=ollama
PROVIDER_API_KEY=
PROVIDER_API_BASE=http://127.0.0.1:11434
PROVIDER_MODEL=qwen3.5:4b
PROVIDER_REQUEST_TIMEOUT_SECONDS=60
```

### 工具结果图片

`ToolMessage.content` 仍是文本；它额外支持一个可选的 `image_path`，表示工具结果关联的一张本地图片路径。核心消息与会话数据不保存图片二进制数据。仅原生 `OllamaCompactProvider` 会在实际请求前读取该路径所指向的单个本地文件，并通过 Ollama SDK 的图片字段发送；`OpenAICompatProvider` 不支持 `ToolMessage.image_path`，遇到该字段会明确拒绝请求，而不会静默丢弃或转换为 OpenAI-compatible 请求。当前不支持多张图片、远程 URL、直接传入二进制数据或默认测试中的真实 VLM/Ollama 调用。

### 内建 `capture_camera` 工具

`capture_camera` 是无参数的内建工具，会返回 `./workspace/pictures/test.jpg` 作为一张本地工具结果图片路径。应用装配时会通过 `ToolLoader` 自动注册该工具。当前实现不读取文件、不连接相机，也不控制机器人硬件；它仅为现有的 Tool → ToolMessage → 原生 Ollama 图片链路提供约定的本地样例路径。实际由 Ollama Provider 发送图片时，该文件必须存在且可读取。

仓库根目录的 `workspace/` 用于本地会话数据，已被 Git 忽略，不应提交其中的内容。

`API_HOST` 默认值为 `127.0.0.1`，避免在未设置认证和网络边界时默认向局域网暴露服务；`API_PORT` 默认值为 `8000`；`API_REQUEST_TIMEOUT_SECONDS` 是 `/v1/messages` 等待关联 Agent 出站消息的上限，不是 Provider SDK 超时，也不适用于同步的会话清空。`PROVIDER_REQUEST_TIMEOUT_SECONDS` 是 Provider SDK 的单次请求超时，默认值为 `60` 秒。所有字段均遵循“进程环境变量优先于根目录 `.env`”的规则。

## 本地 HTTP API 与 CLI

入口使用根目录 `.env` 中的 `AgentConfig` 启动：

```powershell
uv run robot-agent
uv run python -m src
```

`aiohttp` 已加入运行时依赖并同步到 `uv.lock`；`python -m src --help` 与已安装的 `robot-agent --help` 已在本机验证可进入 CLI。HTTP API 不提供认证，仅适用于保持默认本地监听的开发场景；在认证和网络边界措施完成前，不应把 `API_HOST` 改为可被局域网或公网访问的地址。

| 方法 | 路径 | 当前源码行为 |
| --- | --- | --- |
| `GET` | `/health` | 返回服务健康状态。 |
| `POST` | `/v1/messages` | 接收 `{ "session_id": "...", "content": "..." }`，发布当前 `InboundMessage` 并等待同一请求对应的 `OutboundMessage`。不再接受或使用 `channel`、`chat_id`、`sender_id`。 |
| `POST` | `/v1/sessions/clear` | 接收 `{ "session_id": "..." }`，直接通过 `SessionManager` 清空并保存该会话的消息历史，同时保留会话标识。不存在的会话会成为一个空会话。 |
| `GET` | `/v1/sessions` | 返回已持久化会话摘要。 |
| `GET` | `/v1/sessions/{session_id}` | 返回单个会话中可见的用户与助手消息。 |

当前 `HttpApiService` 是共享 `MessageBus` 出站队列的唯一消费者；未来接入其他传输层前，需要先将总线扩展为按订阅者或路由分发，不能直接新增第二个全局出站消费者。

## 测试与文档

默认测试覆盖统一 Agent 配置加载、Provider 工厂构造、Tool 基础设施、非流式 AgentRunner、ContextBuilder、MessageBus/AgentLoop、HTTP/CLI 以及 Session/JSONL 生命周期行为；Ollama 图片传输使用 Mock/Fake SDK Client 与临时本地文件验证，不连接真实 Provider、网络服务、GPU 或设备。HTTP/CLI 测试只使用 Mock/Fake Provider 和内存总线。当前开发进度记录在 [docs/development-progress.md](docs/development-progress.md)。

后续开发必须遵守 [AGENTS.md](AGENTS.md)：先阅读相关文件，保持职责清晰，避免不必要抽象，并使代码、配置、测试和文档保持一致。

## 当前未实现或未验证内容

- Provider 的真实服务联调与端到端集成测试；
- 面向真实视觉模型的 Ollama/VLM 端到端验证；
- 流式 Agent 执行、目标模式、消息注入和运行时调度；
- 更多具体内建工具、设备控制、ROS 2 和硬件适配；
- 外部系统连接与通信协议。
