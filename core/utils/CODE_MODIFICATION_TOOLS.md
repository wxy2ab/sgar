# 代码修改工具使用指南

## 概述

`code_fragment_helper.py`和`code_modification_helper.py`提供了一套完整的代码定位和修改工具，适用于任何需要进行代码修改的场景。

## 快速开始

### 安装

这些工具已经集成在`core/utils/`中，直接导入即可使用：

```python
from core.utils import (
    # 代码片段工具
    extract_function_code,
    extract_class_method_code,
    extract_code_fragment,
    replace_code_fragment,
    get_relevant_imports_and_globals,
    format_fragment_prompt,
    # 代码修改工具
    apply_modification_instructions,
    apply_scoped_modification,
    extract_modification_result,
    replace_function_by_ast,
    parse_modify_block
)
```

## 核心功能

### 1. 代码片段提取

#### 提取函数代码

```python
code = """
import pandas as pd

def calculate_alpha(data, config):
    '''计算alpha因子'''
    result = data.rolling(20).mean()
    return result

def normalize_alpha(alpha):
    return alpha / alpha.std()
"""

# 提取calculate_alpha函数
fragment_info = extract_function_code(
    code=code,
    function_name="calculate_alpha",
    include_context=True,
    context_lines=5
)

# 返回结果
{
    'function_code': 'def calculate_alpha(data, config):\n    ...',
    'start_line': 3,      # 函数起始行号（1-based）
    'end_line': 6,        # 函数结束行号
    'context_before': 'import pandas as pd\n',
    'context_after': '\ndef normalize_alpha(alpha):',
    'indentation': '',    # 缩进字符串
    'total_lines': 9      # 总行数
}
```

#### 提取类方法代码

```python
code = """
class AlphaStrategy:
    def __init__(self, config):
        self.config = config
    
    def on_bar(self, bar):
        '''处理bar数据'''
        alpha = self.calculate_alpha(bar)
        return alpha
    
    def calculate_alpha(self, bar):
        return bar.close.pct_change()
"""

# 提取on_bar方法
method_info = extract_class_method_code(
    code=code,
    class_name="AlphaStrategy",
    method_name="on_bar",
    include_context=True,
    context_lines=5
)
```

#### 通用片段提取

```python
# 使用target_scope统一接口
target_scope = {
    'type': 'function',         # 'function' 或 'class_method'
    'name': 'calculate_alpha',  # 函数/方法名
    'context_lines': 10         # 上下文行数
}

fragment_info = extract_code_fragment(code, target_scope)

# 对于类方法
target_scope = {
    'type': 'class_method',
    'class_name': 'AlphaStrategy',
    'name': 'on_bar',
    'context_lines': 10
}

method_info = extract_code_fragment(code, target_scope)
```

### 2. 代码片段替换

```python
# 获取片段信息
fragment_info = extract_function_code(code, "calculate_alpha")

# 新的函数代码
new_function = """
def calculate_alpha(data, config):
    '''改进的alpha计算'''
    # 添加波动率调整
    volatility = data['close'].pct_change().rolling(20).std()
    alpha = data.rolling(20).mean() / volatility
    return alpha
"""

# 替换
modified_code, success, message = replace_code_fragment(
    original_code=code,
    fragment_info=fragment_info,
    new_fragment_code=new_function
)

if success:
    print(f"✅ {message}")
    print(modified_code)
else:
    print(f"❌ {message}")
```

**特性**：
- 自动调整缩进
- 验证语法正确性
- 保持代码结构

### 3. 提取imports和全局常量

```python
code = """
import pandas as pd
import numpy as np
from scipy import stats

# 全局常量
MAX_LOOKBACK = 100
DEFAULT_CONFIG = {'period': 20}

def calculate_alpha(data):
    return data.rolling(DEFAULT_CONFIG['period']).mean()

class Strategy:
    pass
"""

imports_and_globals = get_relevant_imports_and_globals(code)

# 返回
"""
import pandas as pd
import numpy as np
from scipy import stats
MAX_LOOKBACK = 100
DEFAULT_CONFIG = {'period': 20}
"""
```

### 4. 修改指令系统

#### 修改指令格式

```modify
# 修改1: 查找-替换
FIND:
old code here
REPLACE:
new code here

# 修改2: 替换整个函数
REPLACE_FUNCTION: function_name
CODE:
def function_name(...):
    new implementation

# 修改3: 在指定位置后插入
INSERT_AFTER: "import pandas as pd"
CODE:
import numpy as np
from scipy import stats

# 修改4: 在指定位置前插入
INSERT_BEFORE: "class Strategy:"
CODE:
# Helper function
def helper():
    pass
```

#### 应用修改指令

```python
modify_block = """
FIND:
def old_function():
    pass
REPLACE:
def old_function():
    print("new implementation")

INSERT_AFTER: "import pandas as pd"
CODE:
import numpy as np
"""

modified_code, success, message = apply_modification_instructions(
    original_code=code,
    modify_block=modify_block
)

if success:
    print(f"✅ {message}")
    # 输出: ✅ 成功应用2个修改: 替换: def old_function()..., 插入: import pandas as pd... 后
```

### 5. 自动识别LLM输出模式

```python
llm_response = """
Here's the optimized code:

```python
def calculate_alpha(data, config):
    # Improved implementation
    result = data.ewm(span=20).mean()
    return result
```

This implementation uses exponential weighted moving average for better responsiveness.
"""

# 自动识别模式
modification_result = extract_modification_result(llm_response)

# 返回
{
    'mode': 'scoped_function',  # 识别为单函数模式
    'content': 'def calculate_alpha(data, config):\n    ...',
    'function_name': 'calculate_alpha'
}

# 支持的模式：
# - 'scoped_function': 单个函数定义
# - 'modify_instruction': 修改指令块（```modify```）
# - 'full_code': 完整代码（多个定义）
# - 'error': 无法识别
```

### 6. 范围限定修改

```python
original_code = """
import pandas as pd

def calculate_alpha(data, config):
    return data.rolling(20).mean()

def normalize_alpha(alpha):
    return alpha / alpha.std()
"""

new_function = """
def calculate_alpha(data, config):
    '''改进版本'''
    window = config.get('window', 20)
    return data.ewm(span=window).mean()
"""

# 只替换calculate_alpha函数
modified_code, success, message = apply_scoped_modification(
    original_code=original_code,
    function_name="calculate_alpha",
    new_function_code=new_function
)

# 其他函数（normalize_alpha）保持不变
```

### 7. AST级别函数替换

```python
# 最底层的AST替换函数
modified_code, success = replace_function_by_ast(
    code=original_code,
    function_name="calculate_alpha",
    new_function_code=new_function
)

# 特性：
# - 使用AST精确定位函数位置
# - 自动调整缩进
# - 验证语法
# - 保持其他代码不变
```

## 完整工作流示例

### 示例1：LLM驱动的代码优化

```python
from core.utils import (
    extract_code_fragment,
    get_relevant_imports_and_globals,
    extract_modification_result,
    apply_scoped_modification,
    apply_modification_instructions
)

# 步骤1：提取要优化的代码片段
target_scope = {
    'type': 'function',
    'name': 'calculate_alpha',
    'context_lines': 10
}

fragment_info = extract_code_fragment(original_code, target_scope)
imports = get_relevant_imports_and_globals(original_code)

# 步骤2：构造LLM prompt（只传递片段）
prompt = f"""
Optimize this function:

Imports:
{imports}

Current implementation:
{fragment_info['function_code']}

Only return the optimized function.
"""

# 步骤3：调用LLM
llm_response = await llm.generate(prompt)

# 步骤4：自动识别LLM输出模式
modification_result = extract_modification_result(llm_response)

# 步骤5：应用修改
if modification_result['mode'] == 'scoped_function':
    modified_code, success, message = apply_scoped_modification(
        original_code,
        modification_result['function_name'],
        modification_result['content']
    )
elif modification_result['mode'] == 'modify_instruction':
    modified_code, success, message = apply_modification_instructions(
        original_code,
        modification_result['content']
    )
elif modification_result['mode'] == 'full_code':
    modified_code = modification_result['content']
    success = True

# 步骤6：保存结果
if success:
    with open(file_path, 'w') as f:
        f.write(modified_code)
    print(f"✅ {message}")
```

### 示例2：批量函数重构

```python
# 重构多个函数
functions_to_refactor = [
    'calculate_alpha',
    'normalize_alpha',
    'filter_signals'
]

for func_name in functions_to_refactor:
    # 提取函数
    fragment_info = extract_function_code(
        code=original_code,
        function_name=func_name
    )
    
    if not fragment_info:
        print(f"⚠️  Function {func_name} not found")
        continue
    
    # 重构逻辑（这里可以是LLM或规则）
    new_function = refactor_function(fragment_info['function_code'])
    
    # 替换
    modified_code, success, message = replace_code_fragment(
        original_code,
        fragment_info,
        new_function
    )
    
    if success:
        original_code = modified_code  # 更新以便下次迭代
        print(f"✅ Refactored {func_name}")
```

### 示例3：智能代码更新

```python
# 根据changelog自动更新代码
changelog = """
1. 添加logging支持
2. 优化calculate_metrics函数的性能
3. 修复process_data中的bug
"""

# 生成修改指令
modify_instructions = f"""
INSERT_AFTER: "import pandas as pd"
CODE:
import logging
logger = logging.getLogger(__name__)

REPLACE_FUNCTION: calculate_metrics
CODE:
def calculate_metrics(data):
    logger.info("Calculating metrics")
    # 使用向量化操作提升性能
    return data.apply(lambda x: x.mean(), axis=1)

FIND:
def process_data(df):
    return df.dropna()
REPLACE:
def process_data(df):
    # 修复：检查空DataFrame
    if df.empty:
        logger.warning("Empty dataframe")
        return df
    return df.dropna()
"""

# 应用修改
modified_code, success, message = apply_modification_instructions(
    original_code,
    modify_instructions
)
```

## 最佳实践

### 1. 错误处理

```python
try:
    fragment_info = extract_code_fragment(code, target_scope)
    
    if not fragment_info:
        # 提取失败，使用fallback
        logger.warning("Fragment extraction failed, using full code mode")
        # fallback logic
    
    modified_code, success, message = replace_code_fragment(
        original_code, fragment_info, new_code
    )
    
    if not success:
        logger.error(f"Replacement failed: {message}")
        # 回滚或重试
        
except SyntaxError as e:
    logger.error(f"Syntax error: {e}")
    # 代码有语法错误，不进行修改

except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # 其他错误处理
```

### 2. 验证修改结果

```python
import ast

def validate_modification(original_code, modified_code):
    """验证修改后的代码"""
    
    # 1. 语法验证
    try:
        ast.parse(modified_code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    # 2. 结构验证（确保没有丢失重要部分）
    original_tree = ast.parse(original_code)
    modified_tree = ast.parse(modified_code)
    
    original_functions = {
        node.name for node in ast.walk(original_tree)
        if isinstance(node, ast.FunctionDef)
    }
    
    modified_functions = {
        node.name for node in ast.walk(modified_tree)
        if isinstance(node, ast.FunctionDef)
    }
    
    # 检查是否有函数被意外删除
    missing_functions = original_functions - modified_functions
    if missing_functions:
        return False, f"Missing functions: {missing_functions}"
    
    return True, "Validation passed"

# 使用
modified_code, success, message = apply_scoped_modification(...)
if success:
    valid, validation_message = validate_modification(original_code, modified_code)
    if valid:
        save_code(modified_code)
    else:
        logger.error(f"Validation failed: {validation_message}")
```

### 3. 日志和监控

```python
from core.utils.log import logger

# 详细的日志记录
logger.info(f"📋 Extracting function: {function_name}")
fragment_info = extract_function_code(code, function_name)

if fragment_info:
    logger.info(f"✅ Extracted {fragment_info['end_line'] - fragment_info['start_line']} lines")
else:
    logger.error(f"❌ Failed to extract function: {function_name}")

# 修改统计
logger.info(f"🔧 Applying modification mode: {modification_result['mode']}")

if success:
    logger.info(f"✅ {message}")
else:
    logger.error(f"❌ {message}")
```

### 4. Token优化

```python
# Before: 传递完整代码
prompt = f"Optimize this code:\n{complete_code}"  # 65K tokens

# After: 只传递相关片段
fragment_info = extract_code_fragment(complete_code, target_scope)
imports = get_relevant_imports_and_globals(complete_code)

prompt = f"""
Imports:
{imports}  # 500 tokens

Function to optimize:
{fragment_info['function_code']}  # 2K tokens

Context:
{fragment_info['context_before'][:500]}  # 500 tokens
"""
# Total: 3K tokens (节省95%!)
```

## 工具组合

这些工具可以灵活组合使用：

```python
# 组合1：提取 + 修改 + 验证
fragment = extract_code_fragment(code, scope)
modified, success, msg = replace_code_fragment(code, fragment, new_code)
valid, validation_msg = validate_modification(code, modified)

# 组合2：批量提取 + LLM优化
fragments = [extract_code_fragment(code, scope) for scope in scopes]
optimized_fragments = [await llm_optimize(f) for f in fragments]
final_code = apply_all_modifications(code, fragments, optimized_fragments)

# 组合3：分析 + 指令生成 + 应用
analysis = analyze_code(code)
instructions = generate_instructions(analysis)
modified_code, success, msg = apply_modification_instructions(code, instructions)
```

## 总结

这套代码修改工具提供了：

1. ✅ **精确定位**：AST级别的代码片段提取
2. ✅ **智能替换**：自动调整缩进和验证语法
3. ✅ **灵活模式**：支持多种修改方式
4. ✅ **自动识别**：智能识别LLM输出格式
5. ✅ **Token优化**：大幅减少LLM输入输出
6. ✅ **错误处理**：完善的错误处理和回滚机制

适用场景：
- 代码优化和重构
- LLM驱动的代码生成
- 自动化代码更新
- 批量代码修改
- 代码质量提升

已经在`opt_time_factor`和`construction`系统中得到验证和应用。

