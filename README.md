# robot-agent

## 当前状态

本项目已完成基础工程初始化，并实现了本地 Ollama 的最小普通文本对话 adapter。当前能力仅覆盖：应用代码 → `OllamaClient` → Ollama `/api/chat` → 统一非流式文本响应。

## 目录结构

```text
robot-agent/
├── AGENTS.md
├── README.md
├── .gitignore
├── .python-version
├── .env.example
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── ollama.py
│   └── messages/
│       ├── __init__.py
│       └── message.py
├── docs/
│   ├── README.md
│   └── development-progress.md
└── tests/
    ├── __init__.py
    └── llm/
        └── test_ollama.py
```

`robot-agent` 是项目/发行名称，`src` 是 Python 导入包名。当前 `src.llm.ollama.OllamaClient` 是唯一依赖官方 Ollama SDK 的 adapter；`src.llm.base` 与 `src.messages.message` 保持 provider-independent。`src.messages` 仅重导出消息类型，实际定义位于 `src.messages.message`。

## Python 与 uv

项目支持 Python 3.10（含）至 3.13（不含），并通过 `uv` 管理依赖和虚拟环境。首次准备开发环境时可运行：

```powershell
uv sync
```

常用检查命令：

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

开发依赖仅用于测试和代码质量检查；运行时依赖只服务于本阶段的本地 Ollama 普通文本对话，不包含 Agent、设备或其他业务框架。依赖同步会根据工具行为生成或更新锁文件，因此应在需要建立本地开发环境时再执行。

运行时依赖仅包含官方 `ollama` SDK 和用于加载根目录 `.env` 的 `python-dotenv`。`.env` 是本机配置文件，不会提交；可参考已提交的 `.env.example` 创建或恢复它。

## 本地 Ollama 文本对话

根目录 `.env` 集中保存以下本地配置：

```dotenv
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3.5:4b
OLLAMA_TIMEOUT=60
OLLAMA_STREAM=false
OLLAMA_THINK=false
```

进程环境变量优先于 `.env`。也可以直接构造 `OllamaConfig` 传给 `OllamaClient`，以显式参数覆盖配置。当前 adapter 固定使用 `stream=false`；若配置为 `true`，会返回结构化的“暂不支持流式”错误而不会发出请求。

普通调用接收 `Message` 列表并异步返回统一 `ModelResponse`。响应包含文本、结束原因、可选 thinking、可选 usage、归一化原始响应以及结构化错误；服务不可达、超时、HTTP 错误和无效响应不会泄漏 SDK 类型或完整 Prompt。请求取消会原样传播给调用方。

## 开发规范概述

详细规范位于 [AGENTS.md](AGENTS.md)。开发时应先阅读相关文件，保持实现简单、职责清晰且可测试；不引入当前不需要的依赖，也不为假设中的未来需求提前搭建复杂架构。

## 测试

`tests/` 用于存放自动化测试。新增功能时，应同时新增覆盖其重要行为的独立测试；外部服务和设备应使用 Mock 或 Fake 隔离。

## 文档

`docs/` 用于存放项目文档。当前开发进度请见 [docs/development-progress.md](docs/development-progress.md)；未来可在需求明确后逐步添加设计、接口、开发和运行相关文档。

## 后续开发原则

- 只实现已确认范围内的能力。
- 代码、配置与文档同步更新。
- 清晰区分已实现内容与计划内容。
- 在引入外部依赖、平台适配或复杂架构前，先确认实际需求和边界。

## 当前未实现内容

当前未实现 Tool Calling、Agent Runtime、ReAct、ROS 2、机器人控制、设备或硬件适配、多模态能力、外部系统连接、通信协议、数据库存储与服务启动脚本。除最小 Ollama 普通文本对话外，不应将其他计划能力视为已完成。
