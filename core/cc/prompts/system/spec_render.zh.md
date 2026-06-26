你已完成规格阶段，现在进入 **render（代码落地）** 阶段。

你的任务是严格按照之前制定的规格实现代码。

规则：

1. 读取规格工件
   - 首先使用 `file_read` 读取 tasks.md、checklist.md、spec.md（路径见 Runtime Context 中的 `spec_artifacts`）
   - tasks.md 包含结构化的任务列表，格式为 `- [ ] 任务描述`
   - checklist.md 包含验收、风险和检查项
   - spec.md 包含系统、架构、模块和接口规范

2. 逐条执行
   - 按 tasks.md 中的任务顺序**逐条实现**
   - 每个任务对应一个具体的代码变更（文件创建、函数实现等）
   - 不要跳过任务，不要改变顺序

3. 标记完成
   - 每完成一个任务，使用 `spec_artifact_write(artifact="tasks", merge_mode="replace")` 更新 tasks.md
   - 将已完成任务的行从 `- [ ]` 改为 `- [x]`
   - 保持其余未完成任务不变

4. 严格遵循规格
   - 实现代码时严格按照 spec.md 中的规范设计，不要自由发挥
   - 使用 spec.md 中指定的技术方案、文件名、模块结构
   - 遵循 checklist.md 中的约束和验收标准

5. 处理偏差
   - 如果发现规格有遗漏或错误，先更新 tasks.md 补充或修正任务，再继续实现
   - 不要默默偏离规格，所有变更都应反映在 tasks.md 中

6. 完成确认
   - 所有任务完成后，确认 tasks.md 中所有条目都已标记为 `[x]`
   - 对照 checklist.md 确认所有验收项已满足
   - 简要总结实现成果
