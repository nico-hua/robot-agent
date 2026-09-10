# 开发进度

## 当前开发阶段

当前处于“最小本地文本对话”阶段：应用代码可通过统一 LLM 接口调用本地 Ollama 的 `/api/chat`，并获得统一的非流式文本响应。

## 已完成功能

### 2026-09-10

- 建立了最小的 `Message`、`LLMClient`、`ModelResponse` 和结构化 `ModelError` 抽象；核心接口不暴露 Ollama SDK 类型。
- 新增 `OllamaConfig`，支持根目录 `.env`、进程环境变量和显式配置对象；默认地址为 `http://127.0.0.1:11434`，默认模型为 `qwen3.5:4b`。
- 新增异步 `OllamaClient` adapter，固定执行非流式 `/api/chat` 请求，支持 `think` 配置和调用级 timeout。
- 为服务不可达、超时、HTTP 错误、无效响应和请求取消定义了明确行为；不会记录完整 Prompt 或敏感配置。
- 添加官方 `ollama` SDK 与 `.env` 加载依赖。
- 添加 11 个默认离线 Mock 测试，覆盖配置、消息转换、普通文本、thinking、服务不可达、超时、HTTP 错误、无效响应、流式拒绝和取消传播；测试不访问 Ollama、网络、GPU 或设备，且当前均已通过 pytest 与 Ruff 检查。
- 将 `Message` 与 `MessageRole` 的定义迁移到 `src.messages.message`；`src.messages` 只保留兼容性重导出，避免在包初始化文件中放置领域模型定义。

## 待开发功能

- Tool Calling；
- Agent 与 ReAct 流程；
- ROS 2、机器人控制和硬件适配；
- 多模态输入；
- 会话存储、外部系统集成和通信协议。

## 待优化项

- 在真实使用场景明确后，再评估重试、指标、日志策略和模型参数配置边界。
- 在需要流式交互时，单独设计统一的流式响应接口，而不复用当前非流式接口。

## 待解决问题

- 手动联调需要本机 Ollama 服务运行，并已拉取 `qwen3.5:4b`；默认测试不会访问该服务、网络或 GPU。
- 当前阶段未添加手动 Smoke Test，避免把真实本地服务依赖带入默认测试流程。
