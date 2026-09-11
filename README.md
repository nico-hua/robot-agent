# robot-agent

## 当前状态

项目当前处于“统一 Provider 配置、基础 Provider/Tool 构件和最小非流式消息循环”阶段。现有源码包括：

- 基于 Pydantic 的 `ProviderConfig`；
- 从根目录 `.env` 和进程环境变量加载 Provider 配置与本地工作目录的函数；
- OpenAI-compatible Provider 适配器源码；
- 通用消息模型、工具抽象、工具注册表和内建工具发现基础设施。
- 最小持久化会话：仅维护消息历史的 `Session`、JSONL 存储和 `SessionManager`。
- 内存 `MessageBus`、`ContextBuilder` 和队列式 `AgentLoop`：每条消息均构建临时系统提示词、读取会话历史、执行 `AgentRunner`、持久化结果并发布出站回复。

当前包含最小、非流式的 `AgentRunner`：它按轮调用 `provider.chat()`，顺序处理模型请求的工具调用，并在得到最终响应或达到最大迭代次数时结束。

`Session` 不维护摘要、摘要边界或目标状态；旧 JSONL 会话文件中的这些已移除 Header 字段会在读取时忽略，并在下一次保存时移除。

`MessageBus` 的入站队列会被 `AgentLoop` 连续消费，每条消息由受追踪、可取消的后台任务处理；不同会话可并行处理，同一 `session_id` 从读取历史到保存结果始终串行化。每次请求只向 Provider 发送一个临时系统消息；它不会写入 Session。只要 `AgentRunner` 正常返回 `AgentRunResult`，会话历史、当前用户消息和本轮新增的 Assistant/Tool 消息就会按顺序持久化，即使最终回复为空或达到最大迭代边界。Provider 或 AgentRunner 抛出异常时，循环只发布不含内部细节的失败回复，不写入 Session。

尚未实现流式 Agent 执行、目标模式、消息注入、运行时调度、具体内建工具、设备控制、ROS 2 或硬件适配。Provider 的真实服务联调也不会在默认测试中执行。

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
└── tests/
    ├── agent/
    │   ├── test_context.py
    │   ├── test_loop.py
    │   └── test_runner.py
    ├── config/
    │   └── test_loader.py
    ├── providers/
    │   └── test_factory.py
    ├── session/
    │   └── test_session_lifecycle.py
    └── tools/
        ├── test_base.py
        ├── test_tool_loader.py
        └── test_registry.py
```

`robot-agent` 是项目/发行名称，`src` 是 Python 导入包名。当前打包配置包含 `src.agent`、`src.bus`、`src.config`、`src.providers`、`src.session`、`src.tools` 和 `src.tools.builtin`。

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
- `pydantic`：Provider 配置校验；
- `python-dotenv`：根目录 `.env` 加载。

Ruff 的项目行长规则为 100 个字符。`.env` 是本机配置文件，已被 Git 忽略；只能提交 `.env.example`，不能提交密钥或个人端点配置。

## 项目配置

根目录 `.env` 使用项目级 `PROVIDER_*` 变量和 `WORKSPACE_PATH`，而非特定模型服务的变量：

```dotenv
WORKSPACE_PATH=./workspace
PROVIDER_TYPE=openai_compat
PROVIDER_API_KEY=
PROVIDER_API_BASE=https://api.openai.com/v1
PROVIDER_MODEL=your-model-name
PROVIDER_MAX_TOKENS=1024
PROVIDER_TEMPERATURE=0.7
```

当前支持的 `PROVIDER_TYPE` 为：

- `openai_compat`

调用 `src.config.load_provider_config()` 或 `src.config.load_workspace_path()` 时才会读取 `.env`；已存在的进程环境变量优先于 `.env` 值。`PROVIDER_*` 环境变量会被映射为 `ProviderConfig` 的 `type`、`api_key`、`api_base`、`default_model`、`default_max_tokens` 与 `default_temperature`。`WORKSPACE_PATH` 是本地工作目录；未显式传入路径的 `SessionManager()` 会使用它，且相对路径按当前工作目录解析。

对于不需要凭据的本地兼容端点，`PROVIDER_API_KEY` 可以留空；托管服务通常需要由本机用户填写真实 API Key。应用不得记录或输出该值。

仓库根目录的 `workspace/` 用于本地会话数据，已被 Git 忽略，不应提交其中的内容。

## 测试与文档

默认测试覆盖离线配置加载、Provider 工厂构造、Tool 基础设施、非流式 AgentRunner、ContextBuilder、MessageBus/AgentLoop 以及 Session/JSONL 生命周期行为；不连接 Provider 服务、网络、GPU 或设备。当前开发进度记录在 [docs/development-progress.md](docs/development-progress.md)。

后续开发必须遵守 [AGENTS.md](AGENTS.md)：先阅读相关文件，保持职责清晰，避免不必要抽象，并使代码、配置、测试和文档保持一致。

## 当前未实现或未验证内容

- Provider 的真实服务联调与端到端集成测试；
- 流式 Agent 执行、目标模式、消息注入和运行时调度；
- 具体内建工具、设备控制、ROS 2 和硬件适配；
- 外部系统连接、通信协议和服务启动入口。
