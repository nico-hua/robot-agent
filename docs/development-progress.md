# 开发进度

## 当前开发阶段

当前处于“统一 Provider 配置、基础 Provider/Tool 构件与最小非流式 AgentRunner”阶段。项目以 `ProviderConfig` 描述 OpenAI-compatible Provider 的基础连接与模型参数；根目录 `.env` 仅作为本地配置来源。默认测试不连接真实 Provider 服务。

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

## 待开发功能

- Provider 生命周期与真实服务端到端集成测试；
- 具体内建工具及其权限、失败处理和测试；
- 流式 Agent 执行、目标模式、消息注入、运行时调度和任务编排；
- ROS 2、机器人控制和硬件适配；
- 会话存储、外部系统连接、通信协议和服务启动入口。

## 待优化项

- 在实际 Provider 联调需求明确后，补充超时、重试、观测和错误映射策略。
- 在 Provider 具体能力确认后，再决定是否需要 Provider 专属配置。
- 继续为 Provider、Tool 与 AgentRunner 补充独立、无网络的接口测试。

## 待解决问题

- 本地 `.env` 中必须由用户填写与所选 Provider 匹配的 API Base、模型名，以及托管服务所需的 API Key。
- 当前默认测试验证配置加载、Provider 工厂构造、Tool 基础设施与非流式 AgentRunner；不会验证凭据有效性、模型可用性或实际网络连接。
