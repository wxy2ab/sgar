# FeeReader 使用指南

## 概述

`FeeReader` 是一个期货费用数据读取器，提供了缓存机制和云端/本地透明读取功能。它可以替换原有的直接读取 `fee.json` 文件的代码，提供更可靠和高效的数据访问。

## 主要特性

- **缓存机制**：避免重复读取，提高性能（默认缓存1小时）
- **云端/本地透明**：优先从本地读取，失败时自动从OSS获取
- **自动刷新**：定期检查数据更新
- **线程安全**：支持多线程环境使用
- **单例模式**：全局唯一实例，避免重复初始化

## 快速开始

### 基础使用

```python
from core.utils.fee_reader import get_fee_data, get_instrument_fee, get_exchange_fees

# 获取所有费用数据
all_fees = get_fee_data()

# 获取特定合约的费用信息
rb_fee = get_instrument_fee("RB2501")
if rb_fee:
    print(f"螺纹钢费用信息: {rb_fee}")

# 获取特定交易所的所有费用信息
shfe_fees = get_exchange_fees("SHFE")
print(f"上期所合约数量: {len(shfe_fees)}")
```

### 高级使用

```python
from core.utils.fee_reader import FeeReader

# 获取FeeReader实例
fee_reader = FeeReader()

# 强制刷新缓存
fresh_data = fee_reader.get_fee_data(force_refresh=True)

# 查看缓存状态
cache_info = fee_reader.get_cache_info()
print(f"缓存信息: {cache_info}")

# 清除缓存
fee_reader.clear_cache()
```

## API 参考

### 便捷函数

#### `get_fee_data(force_refresh=False)`
获取所有期货费用数据

**参数：**
- `force_refresh` (bool): 是否强制刷新缓存

**返回：**
- `Dict[str, Any]`: 期货费用数据字典

**示例：**
```python
# 使用缓存数据
data = get_fee_data()

# 强制刷新
fresh_data = get_fee_data(force_refresh=True)
```

#### `get_instrument_fee(instrument_id)`
获取指定合约的费用信息

**参数：**
- `instrument_id` (str): 合约代码

**返回：**
- `Optional[Dict[str, Any]]`: 合约费用信息，不存在时返回None

**示例：**
```python
# 获取螺纹钢费用信息
rb_fee = get_instrument_fee("RB2501")
if rb_fee:
    open_fee = rb_fee.get("OpenRatioByMoney", 0)
    close_fee = rb_fee.get("CloseRatioByMoney", 0)
    print(f"开仓费率: {open_fee}, 平仓费率: {close_fee}")
```

#### `get_exchange_fees(exchange)`
获取指定交易所的所有费用信息

**参数：**
- `exchange` (str): 交易所代码（如SHFE、DCE、CZCE等）

**返回：**
- `Dict[str, Any]`: 该交易所的所有合约费用信息

**示例：**
```python
# 获取上期所所有合约费用
shfe_fees = get_exchange_fees("SHFE")
for instrument_id, fee_info in shfe_fees.items():
    print(f"{instrument_id}: {fee_info}")
```

### FeeReader 类方法

#### `get_cache_info()`
获取缓存状态信息

**返回：**
```python
{
    'has_cache': bool,           # 是否有缓存
    'cache_time': str,           # 缓存时间（ISO格式）
    'cache_valid': bool,         # 缓存是否有效
    'cache_size': int,           # 缓存中的合约数量
    'local_file_exists': bool,   # 本地文件是否存在
    'local_file_size': int       # 本地文件大小
}
```

#### `clear_cache()`
清除缓存，下次获取时会重新加载数据

## 数据格式

### 费用数据结构

```python
{
    "RB2501": {
        "InstrumentID": "rb2501",
        "InstrumentName": "螺纹钢2501",
        "Exchange": "SHFE",
        "OpenRatioByMoney": 0.0001,      # 开仓费率（按金额）
        "OpenRatioByVolume": 1.0,        # 开仓费率（按手数）
        "CloseRatioByMoney": 0.0001,     # 平仓费率（按金额）
        "CloseRatioByVolume": 1.0,       # 平仓费率（按手数）
        "CloseTodayRatioByMoney": 0.0001, # 平今费率（按金额）
        "CloseTodayRatioByVolume": 1.0,   # 平今费率（按手数）
        # ... 其他字段
    },
    # ... 其他合约
}
```

## 迁移指南

### 替换直接文件读取

**原来的代码：**
```python
import json

# 直接读取文件
with open('./json/fee.json', 'r', encoding='utf-8') as f:
    fee_data = json.load(f)

# 获取特定合约费用
rb_fee = fee_data.get("RB2501")
```

**迁移后的代码：**
```python
from core.utils.fee_reader import get_fee_data, get_instrument_fee

# 使用FeeReader获取数据
fee_data = get_fee_data()

# 获取特定合约费用
rb_fee = get_instrument_fee("RB2501")
```

### 替换异常处理

**原来的代码：**
```python
try:
    with open('./json/fee.json', 'r', encoding='utf-8') as f:
        fee_data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"读取费用文件失败: {e}")
    fee_data = {}
```

**迁移后的代码：**
```python
from core.utils.fee_reader import get_fee_data

try:
    fee_data = get_fee_data()
except Exception as e:
    print(f"获取费用数据失败: {e}")
    fee_data = {}
```

## 配置说明

### OSS配置

FeeReader需要以下OSS配置（在Config中）：
- `OSS_ACCESS_KEY_ID`: OSS访问密钥ID
- `OSS_ACCESS_KEY_SECRET`: OSS访问密钥Secret

### 缓存配置

- **缓存时长**：默认1小时，可通过修改 `_cache_duration` 调整
- **本地文件路径**：`./json/fee.json`
- **OSS路径**：`core-ctp/fee.json`

## 故障排除

### 常见问题

1. **无法获取数据**
   - 检查本地 `./json/fee.json` 文件是否存在
   - 检查OSS配置是否正确
   - 检查网络连接

2. **数据不是最新的**
   - 使用 `force_refresh=True` 强制刷新
   - 或者调用 `clear_cache()` 清除缓存

3. **性能问题**
   - 检查缓存是否正常工作
   - 避免频繁调用 `force_refresh=True`

### 调试信息

```python
from core.utils.fee_reader import FeeReader

fee_reader = FeeReader()
cache_info = fee_reader.get_cache_info()
print(f"缓存状态: {cache_info}")
```

## 最佳实践

1. **使用便捷函数**：对于大多数用例，使用 `get_fee_data()` 等便捷函数即可
2. **避免频繁刷新**：只在必要时使用 `force_refresh=True`
3. **错误处理**：始终包含异常处理，防止数据获取失败影响主流程
4. **合理使用缓存**：默认的1小时缓存对大多数场景足够

## 更新数据

要更新费用数据，运行：

```bash
# 通过task.py运行
python task.py fee

# 或直接运行
python task/futures/fee.py
```

这将：
1. 从 openctp.cn 下载最新数据
2. 转换为JSON格式
3. 保存到本地 `./json/fee.json`
4. 上传到OSS `core-ctp/fee.json`

已发布的程序将自动从OSS获取最新数据。