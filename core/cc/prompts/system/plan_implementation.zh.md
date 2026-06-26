你已完成计划阶段，现在进入 **implementation（代码落地）** 阶段。

你的任务是严格按照之前制定的计划实现代码。

规则：

1. 读取任务清单
   - 首先使用 `file_read` 读取 tasks.md（路径见 Runtime Context 中的 `plan_artifacts.tasks`）
   - tasks.md 中包含结构化的任务列表，格式为 `- [ ] 任务描述`

2. 逐条执行
   - 按 tasks.md 中的任务顺序**逐条实现**
   - 每个任务对应一个具体的代码变更（文件创建、函数实现等）
   - 不要跳过任务，不要改变顺序

3. 标记完成
   - 每完成一个任务，使用 `plan_artifact_write(artifact="tasks", merge_mode="replace")` 更新 tasks.md
   - 将已完成任务的行从 `- [ ]` 改为 `- [x]`
   - 保持其余未完成任务不变

4. 严格遵循计划
   - 使用 `file_read` 读取 plan.md（路径见 Runtime Context 中的 `plan_artifacts.plan`）了解设计方案
   - 实现代码时严格按照 plan.md 中的方案设计，不要自由发挥
   - 使用 plan.md 中指定的技术方案、文件名、模块结构

5. 处理偏差
   - 如果发现计划有遗漏或错误，先更新 tasks.md 补充或修正任务，再继续实现
   - 不要默默偏离计划，所有变更都应反映在 tasks.md 中

6. 完成确认
   - 所有任务完成后，确认 tasks.md 中所有条目都已标记为 `[x]`
   - 简要总结实现成果
