# 文档目录

此目录用于记录已确认的架构设计、Provider 接口说明、开发指南、运行方式、设计决策和变更记录。

当前已维护 [development-progress.md](development-progress.md)，它记录统一 `AgentConfig`、依赖、最小非流式 AgentRunner、Session/JSONL 持久化、ContextBuilder、MessageBus/AgentLoop、HTTP/CLI 装配与测试的实际状态。HTTP 请求/响应边界、API 配置、会话清空语义和离线测试也应在此目录中持续记录。

`aiohttp` 已作为运行时依赖同步，HTTP/CLI 的离线测试和 CLI 帮助入口已验证。Provider 的真实服务联调、流式执行、运行时调度、自动编排和具体业务流程尚未形成设计文档，也不应被描述为已完成。
