# 技能(Skills)作者指南 · 速查

> 一份带 frontmatter 的 `SKILL.md` 即一个技能;模型通过 `skill` 工具按需加载其正文照做。
> 本文是随库分发的速查版;完整文档见仓库 `core/cc/docs/16_skill_system.md`(默认不随导出)。

## 写一个技能

在工作目录下放:

```
<工作目录>/skills/<技能名>/SKILL.md
```

```markdown
---
name: code-review
description: >
  代码评审清单。当用户要"评审代码 / review diff / 检查改动质量"时使用。
---

# 代码评审清单
1. grep 搜 TODO/FIXME 并统计。
2. 检查测试覆盖与边界条件。
3. 按"文件:行号 — 问题 — 建议"汇总。
```

下次启动会话(cwd 为该目录),`skill` 工具自动出现,可用技能里就有 `code-review`,
模型调用 `skill(name="code-review")` 即拿到正文。**无需写代码、无需注册。**

## 三个技能根(优先级 project > user > repo)

| `source` | 路径 | 用途 |
|---|---|---|
| `repo` | `<cwd>/skills/` | 入库共享技能 |
| `user` | `~/.<root>/skills/` | 用户全局(`<root>` = 项目目录名,与 `setting.ini` 同位) |
| `project` | `<cwd>/.skills/` | 本地覆盖(通常 gitignore),优先级最高 |

同名技能,高优先级根覆盖低优先级。发现规则:递归 `SKILL.md` + 顶层扁平 `*.md`。

## frontmatter

- `name`(可选):缺省 = 目录名(或扁平 `*.md` 的文件名)。建议小写 kebab-case。
- `description`(建议必填):一句"做什么 + 何时触发",**模型靠它选用技能**。支持
  `description: >` 折叠多行(会规整为单行)。

## 随技能携带资源

把脚本/模板放在 `SKILL.md` 同目录,正文用相对路径引用。`skill` 工具返回的
`data.base_dir` 即该目录,模型据此解析。

## 开关

默认开启;**没有任何技能时 `skill` 工具自动隐藏**。要彻底关闭:
`CCConfig(skills_enabled=False)` 或 `setting.ini` 里 `cc_skills_enabled = false`。

## 动态注册(进程内)

```python
from core.cc.skills import load_skill_registry, SkillDefinition
# 或:from core.ccx import load_skill_registry, SkillDefinition

reg = load_skill_registry(cwd)
reg.register(SkillDefinition(
    name="runtime-helper",
    description="运行时注入的技能。",
    content="# Runtime Helper\n\n正文…",
    source="runtime",
))
```

`SkillRegistry.register()` 就是统一的动态接入 API:无需落盘、无需插件框架。

## 公共 API

`from core.cc.skills import SkillDefinition, SkillRegistry, load_skill_registry, skill_roots, discover_skills`
(`core.ccx` 亦再导出这些,外加 `SkillTool`)。

## 常见问题

- **`skill` 工具没出现**:确认至少有 1 个技能被发现(零技能自隐)、`skills_enabled` 未关、
  `SKILL.md` 路径与文件名正确。
- **加载不到某技能**:`skill(name=...)` 的名要与 frontmatter `name`(或缺省目录/文件名)一致;
  未命中会返回错误码 `SK1001` 并列出可用名。
- **改了 `SKILL.md` 不生效**:技能在构建工具注册表时一次性扫描,需新建会话才会重新发现。
