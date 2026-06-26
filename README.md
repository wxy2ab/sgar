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

复制配置模板并填入你的密钥：

```bash
cp setting.ini.template setting.ini
$EDITOR setting.ini
```

所有键也可以用同名大写的环境变量提供（见 `core/utils/config_setting.py`）。
`setting.ini` 含密钥，已在 `.gitignore` 中忽略，请勿提交。

## 使用

```bash
sgar --help                 # 安装后提供的命令行入口
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
