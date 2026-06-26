# 统一日志系统使用说明

统一日志系统 (`unified_logger.py`) 是一个支持多种环境的日志管理解决方案，能够在有无PySide6的环境中正常工作。

## 🎯 设计目标

- **环境兼容性**: 支持CLI、Web、GUI等不同环境
- **可选依赖**: PySide6为可选依赖，不影响其他环境的使用
- **统一接口**: 提供一致的日志API，简化使用
- **灵活配置**: 支持多种日志模式和处理器

## 📋 功能特性

### 日志模式

| 模式 | 说明 | 依赖 | 适用场景 |
|------|------|------|----------|
| `CLI` | 控制台+文件输出 | 无 | 命令行工具、脚本 |
| `WEB` | 控制台+文件输出（优化版） | 无 | Web服务器、API服务 |
| `GUI` | GUI面板+数据库+文件输出 | PySide6 | 桌面GUI应用 |
| `MIXED` | 所有输出方式 | PySide6（可选） | 混合环境 |

### 自动适配机制

- 如果PySide6不可用，自动将GUI模式降级为CLI/WEB模式
- 提供占位符类确保代码不会因缺少PySide6而崩溃
- 智能检测可用功能并相应调整

## 🚀 基本使用

### 导入和初始化

```python
from core.utils.unified_logger import get_logger, set_log_mode, LogMode

# 设置日志模式（可选，默认会自动选择合适的模式）
set_log_mode(LogMode.WEB)  # 或 'web'

# 获取logger
logger = get_logger("my_module")
```

### 记录日志

```python
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

### 环境检测

```python
from core.utils.unified_logger import has_pyside6, get_available_modes

# 检查PySide6是否可用
if has_pyside6():
    print("PySide6可用，支持GUI功能")
else:
    print("PySide6不可用，使用CLI/WEB模式")

# 获取可用的日志模式
available_modes = get_available_modes()
print(f"可用模式: {[mode.value for mode in available_modes]}")
```

## 🔧 不同环境的使用示例

### CLI环境

```python
from core.utils.unified_logger import get_logger, set_log_mode

# CLI环境推荐设置
set_log_mode('cli')
logger = get_logger("cli_app")

logger.info("CLI应用启动")
```

### Web环境

```python
from core.utils.unified_logger import get_logger, set_log_mode

# Web环境推荐设置
set_log_mode('web')
logger = get_logger("web_server")

logger.info("Web服务器启动")
```

### GUI环境（有PySide6）

```python
from core.utils.unified_logger import get_logger, set_log_mode, get_gui_handler

# GUI环境设置
set_log_mode('gui')
logger = get_logger("gui_app")

# 连接GUI日志处理器
gui_handler = get_gui_handler()
if gui_handler:
    def log_callback(log_data):
        # 处理GUI日志显示
        print(f"GUI: {log_data['message']}")
    
    gui_handler.connect_callback(log_callback)

logger.info("GUI应用启动")
```

## 🛠️ 高级配置

### 自定义日志目录

默认日志目录为 `./data/log`，可以通过修改UnifiedLogger实例来更改：

```python
from core.utils.unified_logger import unified_logger

# 修改日志目录
unified_logger._log_dir = "/custom/log/path"
```

### 动态切换模式

```python
from core.utils.unified_logger import set_log_mode, LogMode

# 运行时切换模式
set_log_mode(LogMode.CLI)  # 切换到CLI模式
# ... 一些操作 ...
set_log_mode(LogMode.WEB)  # 切换到WEB模式
```

### 多个Logger

```python
from core.utils.unified_logger import get_logger

# 创建不同的logger
auth_logger = get_logger("auth")
db_logger = get_logger("database")
api_logger = get_logger("api")

auth_logger.info("用户登录")
db_logger.debug("数据库查询")
api_logger.warning("API请求超时")
```

## 📂 输出位置

### 文件输出

- 位置: `./data/log/trading_system_YYYY-MM-DD.log`
- 格式: `时间 - 模块名 - 级别 - 消息`
- 编码: UTF-8

### 控制台输出

- 支持彩色输出（需要colorlog库）
- 如果colorlog不可用，自动降级为普通格式
- 实时显示日志信息

### 数据库输出（GUI模式）

- 位置: `./data/log/trading_system_logs.db`
- 结构: SQLite数据库
- 查询: 支持按级别、时间、模块等条件查询

### GUI输出（GUI模式）

- 通过信号机制发送到GUI组件
- 支持实时日志显示
- 可自定义日志处理回调

## 🚨 错误处理

### 容错机制

1. **PySide6缺失**: 自动使用占位符类，不影响基本功能
2. **数据库错误**: 数据库处理器失败时，其他处理器继续工作
3. **文件权限**: 文件写入失败时，不影响控制台输出
4. **依赖缺失**: colorlog等可选依赖缺失时，使用基本格式化器

### 错误排查

如果遇到问题，可以使用测试脚本检查：

```bash
python test_unified_logger_compatibility.py
```

## 🔄 迁移指南

### 从旧版本迁移

如果您之前直接使用了PySide6相关功能，需要进行以下调整：

**旧代码:**
```python
from core.utils.unified_logger import GuiLogHandler
from PySide6.QtCore import QObject

# 直接使用Qt功能
handler = GuiLogHandler()
handler.log_signal.connect(my_callback)
```

**新代码:**
```python
from core.utils.unified_logger import get_gui_handler, has_pyside6

# 检查PySide6可用性
if has_pyside6():
    handler = get_gui_handler()
    if handler:
        handler.connect_callback(my_callback)  # 兼容方法
else:
    print("GUI功能不可用")
```

### Web项目集成

在Web项目中使用时，建议在启动脚本中设置：

```python
from core.utils.unified_logger import set_log_mode

# 在应用启动前设置
set_log_mode('web')

# 然后在各个模块中正常使用
from core.utils.unified_logger import get_logger
logger = get_logger(__name__)
```

## 🎯 最佳实践

1. **环境特定设置**: 在应用启动时根据环境设置合适的日志模式
2. **模块命名**: 使用模块名或功能名作为logger名称
3. **日志级别**: 生产环境使用INFO及以上级别，开发环境可以使用DEBUG
4. **异常处理**: 在关键代码周围使用适当的日志记录
5. **性能考虑**: 大量日志输出时考虑使用异步处理

## 📝 示例项目

参考项目中的使用示例：

- **CLI工具**: `alpha/` 目录下的各种脚本
- **Web服务**: `web/server/run_server.py`
- **GUI应用**: `gui/main_window.py`

## 🐛 故障排除

### 常见问题

1. **导入错误**: 确保项目根目录在Python路径中
2. **权限问题**: 确保日志目录有写入权限
3. **编码问题**: 确保系统支持UTF-8编码
4. **依赖冲突**: 检查PySide6版本兼容性

### 调试模式

启用调试日志查看内部工作情况：

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)

from core.utils.unified_logger import get_logger
logger = get_logger("debug_test")
logger.debug("调试信息")
``` 