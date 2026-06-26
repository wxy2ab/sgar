# sgar

`ccx` 治理化 agent 运行时，从 `llm_dealer` 仓库中按 `core/ccx` 的依赖闭包自动导出的
独立、可 pip 安装项目。

本目录由 `task/copy/ccx_out.py` 生成：包含 321 个源文件、61 个运行期资源文件，
覆盖子包 `core.cc`, `core.ccx`, `core.deepstack_v5`, `core.llms`, `core.utils`。

> 注意：分发名是 `sgar`，但可导入的顶层包是 `core`（保留原命名空间以免改写大量
> `import core.*`）。

## 安装

```bash
pip install -e .
# 或
pip install .
```

需要 Python >=3.12。

## 配置

推荐直接使用 `sgar config` 写入用户级配置文件：

```bash
sgar config where
sgar config list
sgar config set --client SimpleDeepSeekClient --api-key YOUR_KEY --model deepseek-v4-pro
```

上述命令会把配置写入 `~/.sgar/setting.ini`，默认同时设置：

```ini
[Default]
llm_api = SimpleDeepSeekClient
cc_default_llm_client = SimpleDeepSeekClient
```

`sgar config list` 会列出当前分发支持的 `ClientName`、对应 `credential_keys`、
`model_keys` 和一句命令示例。

如果某个 client 只有一个凭证键，可以直接用 `--api-key`。如果某个 client 需要多个
凭证或连接参数（例如某些云厂商 client），请改用可重复参数：

```bash
sgar config set --client SparkClient \
  --key xunfei_spark_api_key=YOUR_KEY \
  --key xunfei_spark_secret_key=YOUR_SECRET \
  --model 4.0Ultra
```

仍然可以手工编辑配置文件；`core/utils/config_setting.py` 会优先读取环境变量，其次读取
项目目录 `setting.ini`，再回退到用户目录 `~/.sgar/setting.ini`。

## 使用

```bash
sgar --help                 # 安装后提供的命令行入口
sgar config --help          # 管理用户级 LLM 配置
sgar config list            # 查看支持的 ClientName / key / model
python -m sgar --help
python -m core.ccx.sgar --help
```

```python
from core.ccx.api import AgentRunRequest  # 程序化入口
```

## 自动发布

- 每次 push 到 `main` 后，GitHub Actions 会自动将 `pyproject.toml` 的 patch 版本加 `1`。
- 版本 bump 成功后，会自动构建并发布到 PyPI。
- 如需手动发布，可在 GitHub Actions 里运行 `Publish sgar to PyPI`，并选择 `pypi` 或 `testpypi`。
- 正式发布使用 GitHub Repository Secret `PYPI_API_TOKEN`。
- 如需手动发布到 TestPyPI，请额外配置 `TEST_PYPI_API_TOKEN`。
