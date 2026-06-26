# CCX Self-Review Prompt

## 任务目标
Review `core/ccx` 代码库，找出架构缺陷、设计问题、代码质量问题和优化空间。只输出分析文档，不修改代码。

## Review 范围
`core/ccx` 全部 Python 代码（134 个文件，~54000 行）。聚焦以下方面：

### 1. 架构与设计
- 模块职责划分是否清晰？是否存在过度耦合？
- v5 Runtime 与 ccx 的接口设计是否合理？
- Agent/Plan/Spec 三种模式的职责边界是否清晰？
- Subagent 递归深度控制是否安全？
- Memory/Recall/Store 的持久化策略是否合理？

### 2. 代码质量
- 是否有 TODO/FIXME/NotImplementedError/bare except/pass？
- 类型注解是否完整？
- 函数/类长度是否合理？
- 重复代码片段？
- 命名是否清晰？

### 3. 并发与线程安全
- asyncio 与 threading 的混用是否安全？
- Spawn/subagent 的生命周期管理是否有泄漏风险？
- 共享状态（metadata inheritance、depth tracking）是否线程安全？

### 4. 错误处理
- 异常传播路径是否清晰？
- 取消/超时处理是否完善？
- 子 agent 失败后的回退策略？

### 5. 可扩展性
- 新增 mode 是否需要修改核心代码？
- 工具注册机制是否灵活？
- 配置系统是否支持动态调整？

### 6. 与 v5 Runtime 的集成
- ToolSpec 转换是否正确？
- NodeSpec 依赖关系构建是否合理？
- SpawnResult 的生命周期管理？
- DAG 执行顺序是否符合预期？

### 7. 已知问题检查
- `api.py` 中单次 LLM-with-tools chat 尚未实现（NotImplementedError fallback）
- `ccx_tool.py` 中 legacy caller 兼容性处理
- `metadata_inheritance.py` 中 spawn depth 的强制限制

## 重点审查文件
1. `api.py` — 主入口，与 cc 的兼容层
2. `runtime.py` — v5 Runtime 装配
3. `agents/ccx_tool.py` — 统一 spawn 工具
4. `agents/subagent.py` — subagent 执行核心
5. `agents/governed_spawn.py` — 受控 spawn 逻辑
6. `agents/metadata_inheritance.py` — 元数据继承
7. `modes/agent.py` / `modes/plan.py` / `modes/spec.py` — 三种模式运行器
8. `memory/store.py` / `memory/recall.py` — 记忆系统

## 输出要求
将分析结果写入 `docs/ccx_self_review_2026-06-16.md`，格式：

```markdown
# CCX Self-Review & Improvement Plan

## 执行摘要（Top 10 关键问题）

## 1. 架构与设计
### 1.1 模块职责分析
### 1.2 接口设计问题
### 1.3 改进思路

## 2. 代码质量
### 2.1 技术债务清单
### 2.2 命名与可读性
### 2.3 改进思路

## 3. 并发与线程安全
### 3.1 asyncio/threading 混用分析
### 3.2 生命周期管理
### 3.3 改进思路

## 4. 错误处理
### 4.1 异常传播路径
### 4.2 取消/超时处理
### 4.3 改进思路

## 5. 可扩展性
### 5.1 Mode 扩展机制
### 5.2 工具注册机制
### 5.3 改进思路

## 6. 与 v5 Runtime 集成
### 6.1 集成质量评估
### 6.2 已知问题
### 6.3 改进思路

## 7. 结论与下一步
```

## 约束
- 只分析，不写代码修改
- 以代码为准
- 如果有不确定的地方，标注为「待确认」
