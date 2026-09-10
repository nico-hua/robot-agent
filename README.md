# robot-agent

## 当前状态

项目当前处于“统一 Provider 配置与基础 Provider/Tool 构件”阶段。现有源码包括：

- 基于 Pydantic 的 `ProviderConfig`；
- 从根目录 `.env` 和进程环境变量加载 Provider 配置的函数；
- OpenAI-compatible Provider 适配器源码；
- 通用消息模型、工具抽象、工具注册表和内建工具发现基础设施。

当前包含最小、非流式的 `AgentRunner`：它按轮调用 `provider.chat()`，顺序处理模型请求的工具调用，并在得到最终响应或达到最大迭代次数时结束。

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
│   │   └── runner.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── schema.py
│   ├── providers/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── messages.py
│   │   ├── openai_compat_provider.py
│   └── tools/
│       ├── base.py
│       ├── context.py
│       ├── loader.py
│       ├── registry.py
│       └── builtin/
└── tests/
    ├── agent/
    │   └── test_runner.py
    ├── config/
    │   └── test_loader.py
    ├── providers/
    │   └── test_factory.py
    └── tools/
        ├── test_base.py
        ├── test_tool_loader.py
        └── test_registry.py
```

`robot-agent` 是项目/发行名称，`src` 是 Python 导入包名。当前打包配置包含 `src.agent`、`src.config`、`src.providers`、`src.tools` 和 `src.tools.builtin`。

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

## 统一 Provider 配置

根目录 `.env` 使用项目级 `PROVIDER_*` 变量，而非特定模型服务的变量：

```dotenv
PROVIDER_TYPE=openai_compat
PROVIDER_API_KEY=
PROVIDER_API_BASE=https://api.openai.com/v1
PROVIDER_MODEL=your-model-name
PROVIDER_MAX_TOKENS=1024
PROVIDER_TEMPERATURE=0.7
```

当前支持的 `PROVIDER_TYPE` 为：

- `openai_compat`

调用 `src.config.load_provider_config()` 时才会读取 `.env`；已存在的进程环境变量优先于 `.env` 值。环境变量会被映射为 `ProviderConfig` 的 `type`、`api_key`、`api_base`、`default_model`、`default_max_tokens` 与 `default_temperature`。数值转换和非法配置由 Pydantic 负责。

对于不需要凭据的本地兼容端点，`PROVIDER_API_KEY` 可以留空；托管服务通常需要由本机用户填写真实 API Key。应用不得记录或输出该值。

## 测试与文档

默认测试覆盖离线配置加载、Provider 工厂构造、Tool 基础设施和非流式 AgentRunner 行为，不连接 Provider 服务、网络、GPU 或设备。当前开发进度记录在 [docs/development-progress.md](docs/development-progress.md)。

后续开发必须遵守 [AGENTS.md](AGENTS.md)：先阅读相关文件，保持职责清晰，避免不必要抽象，并使代码、配置、测试和文档保持一致。

## 当前未实现或未验证内容

- Provider 的真实服务联调与端到端集成测试；
- 流式 Agent 执行、目标模式、消息注入和运行时调度；
- 具体内建工具、设备控制、ROS 2 和硬件适配；
- 会话存储、外部系统连接、通信协议和服务启动入口。
