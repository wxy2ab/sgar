# CC Review Prompt

## 任务目标
Review `core/cc` 代码库（CodeAgent 核心模块），找出架构缺陷、设计问题、代码质量问题和优化空间。只输出分析文档，不修改代码。

## Review 范围
`core/cc` 全部 Python 代码。核心文件包括：

1. `api.py` — 主入口，CodeAgent API
2. `config.py` — 配置系统（32911 行，需重点关注）
3. `structured_flow.py` — 结构化流程控制
4. `plan.py` — 规划模块
5. `runtime.py` — 运行时
6. `engine_factory.py` — 引擎工厂
7. `command_runner.py` — 命令执行器
8. `audit.py` — 审计模块
9. `llm.py` — LLM 客户端
10. `providers.py` — 提供商抽象
11. `tools/` — 工具集
12. `agents/` — Agent 实现
13. `memory/` — 记忆系统
14. `editing/` — 文件编辑系统
15. `conversation/` — 会话管理

## 重点分析维度

### 1. 架构与设计
- 模块职责划分是否清晰？是否存在过度耦合？
- api.py 的同步/异步设计是否合理？
- config.py 32911 行 — 是否过度膨胀？配置分层是否合理？
- 与 ccx 的接口设计是否清晰？（ccx 是 cc 的 drop-in replacement）
- 编辑系统（editing/）的 rollback/validator/facade 分层是否合理？
- 记忆系统（memory/）的抽象是否足够？

### 2. 代码质量
- 是否有 TODO/FIXME/NotImplementedError/bare except/pass？
- 函数/类长度是否合理？
- 重复代码片段？
- 命名是否清晰？
- 类型注解是否完整？

### 3. 配置系统（重点）
- config.py 32911 行 — 结构分析
- 是否有配置冗余或冲突？
- 配置验证机制是否完善？
- 默认值是否合理？
- 环境变量覆盖机制是否安全？

### 4. 安全与审计
- audit.py 的审计覆盖是否完整？
- 命令执行（command_runner.py）是否有注入风险？
- 文件编辑（editing/）是否有安全边界？
- safety/ 目录的安全策略是否完备？

### 5. 与 ccx 的关系
- ccx 作为 cc 的 drop-in replacement，cc 的接口是否稳定？
- 哪些 cc 功能被 ccx 替代，哪些保留？
- 两者共存时的状态一致性？

## 输出要求
将分析结果写入 `docs/cc_review_2026-06-16.md`，格式：

```markdown
# CC (CodeAgent) Review & Improvement Plan

## 执行摘要（Top 10 关键问题）

## 1. 架构与设计
### 1.1 模块职责分析
### 1.2 接口设计评估
### 1.3 改进思路

## 2. 代码质量
### 2.1 技术债务清单
### 2.2 命名与可读性
### 2.3 改进思路

## 3. 配置系统深度分析（config.py）
### 3.1 结构分析
### 3.2 发现的问题
### 3.3 改进思路

## 4. 安全与审计
### 4.1 审计覆盖
### 4.2 命令执行安全
### 4.3 文件编辑安全
### 4.4 改进思路

## 5. 与 ccx 的关系
### 5.1 接口稳定性
### 5.2 功能重叠与分离
### 5.3 改进思路

## 6. 结论与下一步
```

## 约束
- 只分析，不写代码修改
- 以代码为准
- 如果有不确定的地方，标注为「待确认」
