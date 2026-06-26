---
name: sgar
description: Run a governed, state-machine-driven build or maintenance task where completion is decided by machine-checkable exit criteria, not the agent's say-so. Use when you need long-horizon, auditable, won't-lie-about-done execution; each stage advances only when its [check: <shell cmd>] gates pass under hermetic verification. Invoke via `python cli.py <command>`.
---

# SGAR skill

SGAR(State-Governed Agent Regime)把一个自治的构建/维护循环治理起来，使 agent
**无法自证一个虚假的"已完成"**。状态存在 LLM 上下文之外；一个 stage 只有在它的
可机器校验退出准则通过时才被承认推进。

## 何时使用
- 需要无人值守、长程、可审计地推进一个有明确验收标准的任务
- 你希望"完成"由 `[check: <cmd>]` 的退出码裁定，而不是由模型自己宣布
- 需要分阶段推进、每阶段留可核验证据并支持回滚

## 如何调用
```bash
python cli.py <command> [options]
```
首次使用先安装依赖：`pip install -r requirements.txt`。

用户级 LLM 配置可直接写入 `~/.sgar/setting.ini`：

```bash
python cli.py config where
python cli.py config list
python cli.py config set --client SimpleDeepSeekClient --api-key YOUR_KEY --model deepseek-v4-pro
```

如果某个 client 需要多个凭证键，请使用可重复参数 `--key KEY=VALUE`。

## 常用命令
| 命令 | 作用 |
|------|------|
| `init` | 初始化 .sgar 工作区 |
| `status` | 查看项目状态 |
| `set-blueprint` / `set-roadmap` / `set-stage-spec` | 写入治理文档(蓝图/路线图/阶段规格) |
| `validate` | 校验治理文档 |
| `start-stage` | 开始一个阶段 |
| `verify` | 记录某阶段的验证证据 |
| `close-stage` | 关闭一个已验证阶段 |
| `mission ...` | 管理隔离的 mission |
| `config ...` | 管理用户级 LLM 配置（写入 `~/.sgar/setting.ini`） |
| `doctor` | 检测缺失文件/状态不一致 |
| `trace` | 查看 SGAR 操作轨迹 |

完整选项：`python cli.py <command> --help`。

## 退出准则契约
在阶段规格里用 `[check: <shell cmd>]` 声明可机器校验的验收标准，其**退出码即权威**
(0 = 通过)。加 `--run-checks` 后 SGAR 会真正执行这些 check 并拒绝与之矛盾的"通过"；
默认启用 hermetic 执行(剥离 user-site / cwd 注入)，降低 check 被绕过作弊的风险。

## 给调用方 agent 的提示
- 永远给任务一个可自检的目标("让 X check 通过")，而非模糊目标。
- 把 `python cli.py status` / `trace` 的输出当权威进度，而非自己的叙述。
- 某 stage 的 check 过不了就如实报告失败，**不要删改 check 或测试来骗过它**。
