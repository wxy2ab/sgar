# 代码更新系统 - 快速开始

## 🚀 5分钟快速入门

### 最简单的用法

```python
from core.utils import smart_replace_code

# 假设LLM返回了新的函数代码
llm_response = """
```python
def calculate_alpha(data, config):
    # 优化后的实现
    result = data * config['factor']
    return result
```
"""

# 提取代码
from core.utils import extract_modification_result
result = extract_modification_result(llm_response)

# 自动替换
modified_code, success, message = smart_replace_code(
    original_code=your_original_code,
    new_code=result['content']
)

if success:
    print("✅ 替换成功！")
    print(modified_code)
else:
    print(f"⚠️ 替换失败: {message}")
```

---

## 📚 三个核心工具

### 1. 智能替换（推荐）

**最常用，90%+的场景适用**

```python
from core.utils import smart_replace_code

modified_code, success, message = smart_replace_code(
    original_code="...",
    new_code="..."  # LLM生成的新函数
)
```

**优势：**
- 🎯 自动识别要替换什么
- 🔧 自动处理缩进和格式
- ✅ 唯一匹配时成功率95%+

---

### 2. 格式修复

**LLM生成的代码有格式问题？自动修复！**

```python
from core.utils import auto_fix_function_code

fixed_code, success, message = auto_fix_function_code(llm_generated_code)

if success:
    # 格式已修复，可以直接使用
    use_code(fixed_code)
```

**自动修复：**
- 装饰器缩进不一致
- Markdown标记（```python）
- 行尾空白
- 语法错误（尝试修复）

---

### 3. 精确替换（歧义时使用）

**当有多个同名函数/方法时**

```python
from core.utils import replace_function_by_ast

# 替换类方法
modified_code, success = replace_function_by_ast(
    code=original_code,
    function_name='Strategy.calculate_alpha',  # 明确指定类名
    new_function_code=new_code
)
```

---

## 🎯 完整工作流程

### 场景：优化一个函数

```python
from core.utils import (
    extract_modification_result,
    smart_replace_code,
    apply_modification_instructions
)

def optimize_with_llm(original_code: str, llm_client, function_name: str):
    # ===== 步骤1：让LLM生成代码 =====
    prompt = f"""
优化函数: {function_name}

原始代码：
```python
{original_code}
```

只返回优化后的函数，使用```python```标记。
"""
    
    response1 = llm_client.generate(prompt)
    
    # ===== 步骤2：提取并尝试自动替换 =====
    result = extract_modification_result(response1)
    
    if result['mode'] == 'scoped_function':
        modified_code, success, message = smart_replace_code(
            original_code=original_code,
            new_code=result['content']
        )
        
        if success:
            return modified_code  # ✅ 成功！
        
        # ===== 步骤3：如果有歧义，让LLM生成精确指令 =====
        if "存在歧义" in message:
            prompt2 = f"""
存在歧义：
{message}

新代码：
```python
{result['content']}
```

请生成```modify```指令，明确指定要替换哪一个。
"""
            
            response2 = llm_client.generate(prompt2)
            result2 = extract_modification_result(response2)
            
            if result2['mode'] == 'modify_instruction':
                modified_code, success, msg = apply_modification_instructions(
                    original_code=original_code,
                    modify_block=result2['content']
                )
                
                if success:
                    return modified_code  # ✅ 成功！
    
    return original_code  # ❌ 失败，返回原始代码
```

---

## 💡 Prompt模板

### 模板1：生成代码（第一次调用）

```
请优化以下函数：{function_name}

## 原始代码
```python
{original_code}
```

## 要求
{your_requirements}

## 输出格式
只返回优化后的函数代码：

```python
def {function_name}(...):
    # 你的实现
    pass
```

**重要**：
- 不要同时返回修改指令
- 只返回单个函数（不是整个文件）
- 不要包含```modify```块
```

### 模板2：生成修改指令（第二次调用，仅在歧义时）

```
之前的自动替换失败了，因为存在多个同名函数：

{ambiguity_info}

请生成精确的修改指令。

## 新代码
```python
{new_code}
```

## 输出格式
使用```modify```标记，明确指定要替换哪一个：

```modify
REPLACE_FUNCTION: ClassName.method_name
CODE:
{new_code}
```

**说明**：
- 顶层函数：`REPLACE_FUNCTION: function_name`
- 类方法：`REPLACE_FUNCTION: ClassName.method_name`
- 不要同时包含```python```块
```

---

## 🔍 常见问题

### Q1: 什么时候用智能替换？什么时候用精确替换？

**A:** 
- ✅ **优先用智能替换** - 90%+场景成功
- ⚠️ **只在失败时用精确替换** - 特别是存在歧义时

### Q2: LLM返回的代码有格式问题怎么办？

**A:** 
```python
from core.utils import auto_fix_function_code

# 自动修复
fixed_code, success, message = auto_fix_function_code(llm_code)
```

自动修复会处理：
- 装饰器缩进
- Markdown标记
- 空白和空行
- 常见语法错误

### Q3: 如何处理多个同名函数的歧义？

**A:**

**方法1：使用context（如果事先知道）**
```python
modified_code, success, message = smart_replace_code(
    original_code=code,
    new_code=new_code,
    context={
        'target_scope': {
            'type': 'method',
            'name': 'calculate',
            'class': 'Strategy'  # 明确指定类名
        }
    }
)
```

**方法2：让LLM生成精确指令**
```python
# 第二次调用LLM，返回：
```modify
REPLACE_FUNCTION: Strategy.calculate
CODE:
{new_code}
```
```

### Q4: 可以一次修改多个函数吗？

**A:** 可以，但不推荐。

**推荐做法：**
- 分别修改每个函数
- 每次一个函数，更清晰

**如果必须批量修改：**
```python
from core.utils import replace_function_by_ast

for func_name, new_code in modifications.items():
    original_code, success = replace_function_by_ast(
        code=original_code,
        function_name=func_name,
        new_function_code=new_code
    )
```

### Q5: 支持哪些语言？

**A:** 目前只支持Python。

原因：使用Python的AST进行精确分析和替换。

---

## 📖 更多资源

### 详细文档
- [完整工作流程](./CODE_UPDATE_WORKFLOW.md) - 详细的流程说明和示例
- [系统总结](./CODE_UPDATE_SYSTEM_SUMMARY.md) - 设计原则和更新内容
- [工具文档](./CODE_MODIFICATION_TOOLS.md) - API参考

### 示例实现
- `core/task/opt_time_factor/` - 经过验证的实现
- `core/agent/construction/` - Construction系统集成

---

## ✅ 检查清单

在使用之前，确保：

- [ ] 已导入所需工具
  ```python
  from core.utils import smart_replace_code, extract_modification_result
  ```

- [ ] Prompt中明确说明输出格式
  ```python
  "只返回```python```代码块，不要返回```modify```"
  ```

- [ ] 准备好两次LLM调用的流程
  ```python
  # 第1次：生成代码
  # 第2次：生成指令（如果需要）
  ```

- [ ] 有错误处理机制
  ```python
  if not success:
      logger.error(f"失败: {message}")
  ```

---

## 🎓 学习路径

### 初学者
1. 阅读本文档（快速开始）
2. 运行简单示例
3. 尝试优化一个函数

### 进阶用户
1. 阅读 [完整工作流程](./CODE_UPDATE_WORKFLOW.md)
2. 了解修改指令格式
3. 处理复杂场景（歧义、批量等）

### 高级用户
1. 阅读 [系统总结](./CODE_UPDATE_SYSTEM_SUMMARY.md)
2. 研究 opt_time_factor 实现
3. 自定义格式修复规则
4. 集成到自己的Agent

---

## 💬 获取帮助

遇到问题？

1. 查看 [常见问题](#-常见问题)
2. 检查日志中的详细错误信息
3. 参考 [完整工作流程](./CODE_UPDATE_WORKFLOW.md) 的调试部分
4. 查看 opt_time_factor 的实现

---

## 🎉 开始使用

```python
# 最简单的开始方式
from core.utils import smart_replace_code, extract_modification_result

# 1. 获取LLM响应
llm_response = your_llm_client.generate(prompt)

# 2. 提取代码
result = extract_modification_result(llm_response)

# 3. 智能替换
modified_code, success, message = smart_replace_code(
    original_code=your_code,
    new_code=result['content']
)

# 4. 使用结果
if success:
    print("✅ 成功！")
else:
    print(f"⚠️ {message}")
```

就这么简单！🚀

