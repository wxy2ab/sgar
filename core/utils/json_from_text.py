from typing import Union, Dict, List, Any, Optional
import json
import re
from core.utils.log import logger


def extract_json_from_text(
    text: str, 
    llm_client_name: str = "QianWenCoderClient", 
    max_attempts: int = 3,
    use_llm: bool = True
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    从文本中提取JSON对象并返回字典或字典列表。优先使用代码方法，仅在必要时使用LLM。

    Args:
        text (str): 包含JSON数据的字符串
        llm_client_name (str): LLM客户端名称，默认为"QianWenCoderClient"
        max_attempts (int): 最大LLM修复尝试次数，默认为3
        use_llm (bool): 是否允许使用LLM进行修复，默认为True

    Returns:
        Union[Dict[str, Any], List[Dict[str, Any]]]: 解析后的JSON对象

    Raises:
        json.JSONDecodeError: 如果在多次尝试后仍未能找到有效的JSON数据
    """
    
    def _check_brackets_balanced(s: str) -> bool:
        """检查 JSON 字符串的括号是否平衡（考虑字符串内转义）。"""
        depth_brace = 0
        depth_bracket = 0
        in_string = False
        escape = False
        for ch in s:
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth_brace += 1
            elif ch == '}':
                depth_brace -= 1
            elif ch == '[':
                depth_bracket += 1
            elif ch == ']':
                depth_bracket -= 1
        return depth_brace == 0 and depth_bracket == 0

    def _extract_code_fence_robust(text: str, lang_tag: str = "json") -> List[str]:
        """健壮的代码围栏提取，处理 JSON 内容中嵌套 ``` 的情况。

        策略：找到所有 ```<lang> 开头位置，然后找到行首的 ``` 闭合标记，
        优先选择括号平衡的候选。若非贪婪匹配不平衡则回退到贪婪匹配。
        """
        results = []
        open_pattern = re.compile(r'```' + re.escape(lang_tag) + r'\s*\n?')
        close_pattern = re.compile(r'\n\s*```\s*$', re.MULTILINE)

        for open_match in open_pattern.finditer(text):
            content_start = open_match.end()
            remaining = text[content_start:]

            close_positions = [m.start() for m in close_pattern.finditer(remaining)]
            if not close_positions:
                candidate = remaining.strip()
                if candidate:
                    results.append(candidate)
                continue

            # 非贪婪: 第一个闭合
            first_content = remaining[:close_positions[0]].strip()
            if _check_brackets_balanced(first_content):
                results.append(first_content)
                continue

            # 非贪婪不平衡，尝试贪婪: 最后一个闭合
            last_content = remaining[:close_positions[-1]].strip()
            if _check_brackets_balanced(last_content):
                results.append(last_content)
            else:
                results.append(last_content)
            if first_content != last_content:
                results.append(first_content)

        return results

    def extract_json_content(text: str) -> List[str]:
        """从文本中提取所有可能的JSON内容，返回候选列表（按优先级排序，去重）"""
        candidates = []
        seen = set()  # 用于去重
        
        logger.debug(f"开始从文本中提取JSON，文本长度: {len(text)}")
        
        def add_candidate(content: str, source: str) -> bool:
            """添加候选（去重）"""
            if content and content not in seen:
                candidates.append(content)
                seen.add(content)
                logger.debug(f"[{source}] 添加候选，长度: {len(content)}, 前80字符: {content[:80]}")
                return True
            return False
        
        # ✨ 优先级1: 使用健壮的代码围栏提取（处理嵌套 ``` 的情况）
        robust_blocks = _extract_code_fence_robust(text, "json")
        for content in robust_blocks:
            add_candidate(content, "json代码块(健壮)")

        # 回退: 同时保留简单正则匹配作为补充
        json_blocks = list(re.finditer(r'```json\s*([\s\S]*?)\s*```', text))
        logger.debug(f"找到 {len(json_blocks)} 个json标记的代码块(正则), {len(robust_blocks)} 个(健壮)")
        
        # 记录已经提取的文本范围，避免重复
        # Treat all fenced code blocks as already scanned. SQL fences often
        # contain bracket quoting or list-like text that is not JSON.
        extracted_ranges = [
            (match.start(), match.end())
            for match in re.finditer(r'```\w*\s*[\s\S]*?\s*```', text)
        ]
        for match in json_blocks:
            content = match.group(1).strip()
            if add_candidate(content, "json代码块"):
                extracted_ranges.append((match.start(), match.end()))
        
        # 如果找到了json代码块，优先使用，不再尝试其他方法
        if candidates:
            logger.debug(f"已找到json代码块候选，跳过其他提取方法")
            logger.debug(f"总共 {len(candidates)} 个唯一候选（去重后）")
            return candidates
        
        # 优先级2: 其他代码块（```...```，排除已匹配的json块）
        # 使用负向预查排除json标记
        code_blocks = list(re.finditer(r'```(?!json)(\w*)\s*([\s\S]*?)\s*```', text))
        logger.debug(f"找到 {len(code_blocks)} 个其他代码块")
        for match in code_blocks:
            content = match.group(2).strip()
            lang = match.group(1) or '未标记'
            if content and (content.startswith('{') or content.startswith('[')):
                if add_candidate(content, f"{lang}代码块"):
                    extracted_ranges.append((match.start(), match.end()))
        
        # 优先级3: 内联代码块（`...`）
        inline_blocks = list(re.finditer(r'`([^`]+)`', text))
        logger.debug(f"找到 {len(inline_blocks)} 个内联代码块")
        for match in inline_blocks:
            content = match.group(1).strip()
            if content and (content.startswith('{') or content.startswith('[')):
                add_candidate(content, "内联代码块")
        
        # 优先级4: 直接提取JSON对象（支持嵌套，智能括号匹配）
        # ✨ 只在没有找到代码块的情况下才执行
        if not candidates:
            logger.debug("未找到代码块，尝试直接提取JSON对象")
            direct_count = 0
            for start_char, end_char in [('{', '}'), ('[', ']')]:
                pos = 0
                while pos < len(text):
                    start = text.find(start_char, pos)
                    if start == -1:
                        break
                    
                    # 检查是否在已提取的范围内
                    in_extracted = False
                    for range_start, range_end in extracted_ranges:
                        if range_start <= start < range_end:
                            in_extracted = True
                            break
                    
                    if in_extracted:
                        pos = start + 1
                        continue
                    
                    # 寻找匹配的结束括号（考虑字符串和转义）
                    depth = 0
                    end = start
                    in_string = False
                    escape = False
                    
                    for i in range(start, len(text)):
                        char = text[i]
                        
                        if escape:
                            escape = False
                            continue
                        
                        if char == '\\':
                            escape = True
                            continue
                        
                        if char == '"' and not in_string:
                            in_string = True
                        elif char == '"' and in_string:
                            in_string = False
                        elif not in_string:
                            if char == start_char:
                                depth += 1
                            elif char == end_char:
                                depth -= 1
                                if depth == 0:
                                    end = i + 1
                                    break
                    
                    if end > start:
                        if add_candidate(text[start:end], f"直接提取{start_char}"):
                            direct_count += 1
                        pos = end
                    else:
                        pos = start + 1
            
            logger.debug(f"直接提取到 {direct_count} 个JSON对象")
        
        # 如果没有找到任何候选，返回原文本
        if not candidates:
            logger.debug("未找到任何JSON候选，使用原文本")
            candidates.append(text.strip())
        
        # 按长度排序，优先尝试较长的候选（更完整）
        # 但json标记的代码块始终保持在前面（已经按优先级添加）
        # 这里只对后续添加的候选排序
        if len(candidates) > len(json_blocks):
            # 分离json代码块和其他候选
            json_block_count = len(json_blocks)
            json_candidates = candidates[:json_block_count]
            other_candidates = candidates[json_block_count:]
            other_candidates.sort(key=len, reverse=True)
            candidates = json_candidates + other_candidates
        
        logger.debug(f"总共 {len(candidates)} 个唯一候选（去重后）")
        
        return candidates
    
    def clean_json_string(json_str: str) -> str:
        """清理JSON字符串的基本问题"""
        # 删除BOM标记
        json_str = json_str.replace('\ufeff', '')
        
        # 删除单行注释
        json_str = re.sub(r'//[^\n]*', '', json_str)
        
        # 删除多行注释
        json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)
        
        # 移除Python长整型标记
        json_str = re.sub(r'\b(\d+)L\b', r'\1', json_str)
        
        # 移除尾随逗号（在对象和数组中）
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # 合并连续逗号（如 [a,,b] → [a,b]），LLM 常见输出错误
        json_str = re.sub(r',(\s*),+', ',', json_str)
        
        return json_str
    
    def fix_quotes(json_str: str) -> str:
        """修复引号问题"""
        # 将单引号转换为双引号（但要小心字符串内部的单引号）
        # 这是一个简化的实现，对于复杂情况可能不完美
        result = []
        i = 0
        in_double_quote = False
        
        while i < len(json_str):
            char = json_str[i]
            
            if char == '\\' and i + 1 < len(json_str):
                # 转义字符，保持原样
                result.append(char)
                result.append(json_str[i + 1])
                i += 2
                continue
            
            if char == '"':
                in_double_quote = not in_double_quote
                result.append(char)
            elif char == "'" and not in_double_quote:
                # 单引号转双引号（在非字符串上下文中）
                result.append('"')
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)
    
    def fix_unquoted_keys(json_str: str) -> str:
        """修复未加引号的键"""
        # 匹配模式：字母开头的键名（未加引号）后跟冒号
        # 例如: {key: "value"} -> {"key": "value"}
        json_str = re.sub(
            r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)',
            r'\1"\2"\3',
            json_str
        )
        return json_str
    
    def fix_boolean_and_null(json_str: str) -> str:
        """修复布尔值和null值的大小写"""
        # 确保true/false/null是小写
        json_str = re.sub(r'\bTrue\b', 'true', json_str)
        json_str = re.sub(r'\bFalse\b', 'false', json_str)
        json_str = re.sub(r'\bNone\b', 'null', json_str)
        json_str = re.sub(r'\bNull\b', 'null', json_str)
        json_str = re.sub(r'\bNULL\b', 'null', json_str)
        
        # 处理NaN和Infinity（非标准JSON值）
        json_str = re.sub(r'\bNaN\b', 'null', json_str)
        json_str = re.sub(r'\bInfinity\b', '1e308', json_str)
        json_str = re.sub(r'\-Infinity\b', '-1e308', json_str)
        
        return json_str
    
    def fix_escape_sequences(json_str: str) -> str:
        """修复转义序列"""
        # 修复常见的错误转义
        json_str = json_str.replace('\\\n', '\\n')
        json_str = json_str.replace('\\\t', '\\t')
        json_str = json_str.replace('\\\r', '\\r')
        return json_str
    
    def fix_unescaped_strings(json_str: str) -> str:
        """
        智能修复JSON字符串中未转义的特殊字符
        主要处理字符串值中的未转义引号和换行符
        """
        import re
        
        # 正则表达式匹配JSON字符串值："key": "value"
        # 我们需要找到所有字符串值并修复其中的未转义字符
        
        def fix_string_value(match):
            """修复单个字符串值"""
            full_match = match.group(0)
            key_part = match.group(1)  # "key":
            value_content = match.group(2)  # 引号之间的内容
            
            # 修复未转义的引号（但跳过已转义的）
            # 将字符串中的 " 替换为 \"（如果前面没有\）
            fixed_value = re.sub(r'(?<!\\)"', r'\\"', value_content)
            
            # 修复未转义的换行符
            fixed_value = fixed_value.replace('\n', '\\n')
            fixed_value = fixed_value.replace('\r', '\\r')
            fixed_value = fixed_value.replace('\t', '\\t')
            
            # 修复反斜杠（但保留已经转义的）
            # 这个比较复杂，暂时跳过以避免过度修复
            
            return f'{key_part}"{fixed_value}"'
        
        try:
            # 匹配模式: "key": "value"
            # 使用非贪婪匹配，避免跨字段匹配
            pattern = r'("[\w_]+"\s*:\s*)"([^"]*(?:\\.[^"]*)*)"'
            
            # 注意：这个正则表达式可能不完美，对于复杂的嵌套JSON可能有问题
            # 但对于大多数LLM生成的简单JSON应该足够了
            fixed = re.sub(pattern, fix_string_value, json_str, flags=re.DOTALL)
            return fixed
        except Exception as e:
            logger.debug(f"fix_unescaped_strings失败: {e}，返回原字符串")
            return json_str
    
    def fix_over_escaped_json(json_str: str) -> str:
        """修复 LLM 过度转义结构性引号的 JSON。

        LLM 有时会对 JSON 结构性界定符 ``"`` 也加 ``\\`` 转义，
        如 ``[\\\"text\\\"]`` 而非 ``["text"]``。通过上下文判断
        ``\\"`` 是否处于结构性位置（前/后为 ``,`` ``[`` ``{`` ``:``
        ``]`` ``}``），若是则移除多余的 ``\\``。
        """
        result: list[str] = []
        i = 0
        length = len(json_str)
        structural = {',', '[', '{', ':', ']', '}'}

        while i < length:
            if json_str[i] == '\\' and i + 1 < length and json_str[i + 1] == '"':
                if i >= 2 and json_str[i - 1] == '\\':
                    result.append(json_str[i])
                    i += 1
                    continue
                before = ''
                for j in range(i - 1, -1, -1):
                    if not json_str[j].isspace():
                        before = json_str[j]
                        break
                after = ''
                for j in range(i + 2, length):
                    if not json_str[j].isspace():
                        after = json_str[j]
                        break
                is_structural = (before in structural or before == ''
                                 or after in structural or after == '')
                if is_structural:
                    result.append('"')
                    i += 2
                else:
                    result.append('\\')
                    result.append('"')
                    i += 2
            else:
                result.append(json_str[i])
                i += 1

        return ''.join(result)

    def try_repair_truncated(json_str: str) -> Optional[str]:
        """尝试修复被截断的 JSON（补全未闭合的字符串和括号）。"""
        s = json_str.rstrip()
        in_string = False
        escape = False
        bracket_stack = []
        for ch in s:
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ('{', '['):
                bracket_stack.append(ch)
            elif ch == '}':
                if bracket_stack and bracket_stack[-1] == '{':
                    bracket_stack.pop()
            elif ch == ']':
                if bracket_stack and bracket_stack[-1] == '[':
                    bracket_stack.pop()
        if not bracket_stack and not in_string:
            return None
        repaired = s
        if in_string:
            repaired += '"'
        for bracket in reversed(bracket_stack):
            repaired += '}' if bracket == '{' else ']'
        return repaired

    def apply_progressive_fixes(json_str: str) -> List[str]:
        """应用逐步修复策略，返回多个修复后的候选"""
        candidates = []

        # 策略1: 过度转义修复（LLM 对结构引号也加了 \ ）
        step_oe = fix_over_escaped_json(json_str)
        if step_oe != json_str:
            candidates.append(step_oe)

        # 策略2: 基础清理
        step1 = clean_json_string(json_str)
        candidates.append(step1)
        
        # 策略3: 基础清理 + 未转义字符修复（新增，优先级高）
        step2 = fix_unescaped_strings(step1)
        candidates.append(step2)
        
        # 策略3: 基础清理 + 引号修复
        step3 = fix_quotes(step1)
        candidates.append(step3)
        
        # 策略4: 基础清理 + 未转义字符修复 + 键修复
        step4 = fix_unquoted_keys(step2)
        candidates.append(step4)
        
        # 策略5: 基础清理 + 引号修复 + 键修复（原策略3）
        step5 = fix_unquoted_keys(step3)
        candidates.append(step5)
        
        # 策略6: 完整修复（所有修复步骤）
        step6 = fix_boolean_and_null(step5)
        step6 = fix_escape_sequences(step6)
        candidates.append(step6)
        
        # 策略7: 未转义字符修复 + 完整流程
        step7 = fix_boolean_and_null(step4)
        step7 = fix_escape_sequences(step7)
        candidates.append(step7)
        
        # 策略8: 尝试直接修复未加引号的键（跳过引号转换）
        step8 = fix_unquoted_keys(clean_json_string(json_str))
        step8 = fix_boolean_and_null(step8)
        candidates.append(step8)
        
        # 策略9: 截断修复 — 补全未闭合的字符串和括号
        repaired = try_repair_truncated(step1)
        if repaired:
            candidates.append(repaired)
            repaired2 = try_repair_truncated(step2)
            if repaired2:
                candidates.append(repaired2)
        
        return candidates
    
    def try_parse_json(json_str: str) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """尝试解析JSON字符串，返回None如果失败"""
        try:
            result = json.loads(json_str)
            # 验证结果类型
            if isinstance(result, (dict, list)):
                logger.debug(f"✓ JSON解析成功，类型: {type(result).__name__}, 键数/长度: {len(result)}")
                return result
            else:
                logger.debug(f"✗ JSON解析结果类型无效: {type(result).__name__}")
        except json.JSONDecodeError as e:
            logger.debug(f"✗ JSON解析失败: {e}")
        except (ValueError, TypeError) as e:
            logger.debug(f"✗ JSON验证失败: {type(e).__name__}: {e}")
        return None
    
    def llm_fix_json(json_str: str, error_msg: str) -> str:
        """使用LLM修复JSON（仅在代码方法全部失败时使用）"""
        try:
            from core.llms.llm_factory import LLMFactory
            logger.info(f"尝试使用LLM ({llm_client_name}) 修复JSON...")
            logger.info(f"待修复JSON长度: {len(json_str)} 字符")
            llm_client = LLMFactory().get_instance(llm_client_name)
            
            prompt = f"""请修复以下JSON字符串，确保返回有效的JSON格式。

错误信息: {error_msg}

原始JSON:
```
{json_str}
```

修复要求:
1. 只返回修复后的JSON，不要有任何解释文字
2. 保持原始数据结构和值不变
3. 修复所有语法错误：未加引号的键、多余逗号、引号不匹配等
4. 确保布尔值是小写(true/false)，null是小写
5. 不要添加代码块标记(```json)
6. 对于长JSON，请仔细检查所有括号是否配对、所有字符串是否正确闭合
7. 不要省略或截断任何内容，完整返回修复后的JSON

直接返回修复后的JSON:"""
            
            response = llm_client.one_chat(prompt)
            
            # 从LLM响应中提取JSON
            llm_candidates = extract_json_content(response)
            if llm_candidates:
                return llm_candidates[0]
            return response.strip()
            
        except Exception as e:
            logger.warning(f"LLM修复JSON失败: {e}")
            return json_str
    
    # ========== 主处理逻辑 ==========
    
    logger.debug("=" * 60)
    logger.debug("开始JSON提取流程")
    logger.debug("=" * 60)
    
    # 步骤1: 提取所有可能的JSON候选
    json_candidates = extract_json_content(text)
    logger.info(f"步骤1: 提取到 {len(json_candidates)} 个JSON候选")
    
    # 步骤2: 对每个候选应用代码级修复并尝试解析
    logger.info(f"步骤2: 开始尝试解析候选...")
    for i, candidate in enumerate(json_candidates, 1):
        if not candidate:
            continue
        
        logger.debug(f"--- 尝试候选 {i}/{len(json_candidates)} (长度: {len(candidate)}) ---")
        
        # 首先尝试直接解析
        logger.debug("  策略0: 直接解析")
        result = try_parse_json(candidate)
        if result is not None:
            logger.info(f"✓ 候选 {i} 直接解析成功！")
            return result
        
        # 应用逐步修复策略
        fixed_candidates = apply_progressive_fixes(candidate)
        logger.debug(f"  策略1-{len(fixed_candidates)}: 应用修复策略")
        
        for j, fixed in enumerate(fixed_candidates, 1):
            logger.debug(f"    尝试修复策略 {j}/{len(fixed_candidates)}")
            result = try_parse_json(fixed)
            if result is not None:
                logger.info(f"✓ 候选 {i} 使用修复策略 {j} 成功！")
                return result
    
    # 步骤2.5: 预处理原文——修复过度转义后重新提取候选
    fixed_text = fix_over_escaped_json(text)
    if fixed_text != text:
        logger.info("步骤2.5: 检测到过度转义，尝试修复后重新提取...")
        result = try_parse_json(fixed_text)
        if result is not None:
            logger.info("✓ 过度转义修复后直接解析成功！")
            return result
        re_candidates = extract_json_content(fixed_text)
        for idx, rc in enumerate(re_candidates, 1):
            result = try_parse_json(rc)
            if result is not None:
                logger.info(f"✓ 过度转义修复后候选 {idx} 解析成功！")
                return result

    # 步骤3: 如果代码方法全部失败，且允许使用LLM，则尝试LLM修复
    if use_llm and max_attempts > 0:
        logger.info(f"步骤3: 代码方法全部失败，尝试LLM修复...")
        
        # 使用第一个（最长的）候选进行LLM修复
        best_candidate = json_candidates[0] if json_candidates else text
        
        last_error = ""
        try:
            json.loads(best_candidate)
        except json.JSONDecodeError as e:
            last_error = str(e)
            logger.debug(f"最后的错误信息: {last_error}")
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"--- LLM修复尝试 {attempt + 1}/{max_attempts} ---")
                fixed_json = llm_fix_json(best_candidate, last_error)
                
                # 尝试解析LLM修复的结果
                result = try_parse_json(fixed_json)
                if result is not None:
                    logger.info("LLM成功修复JSON")
                    return result
                
                # 如果LLM返回的结果还是无法解析，再次应用代码级修复
                for fixed in apply_progressive_fixes(fixed_json):
                    result = try_parse_json(fixed)
                    if result is not None:
                        return result
                
                # 更新候选为LLM修复的版本
                best_candidate = fixed_json
                
            except Exception as e:
                logger.warning(f"LLM修复尝试 {attempt + 1} 失败: {e}")
                continue
    
    # 所有方法都失败了 - 生成详细的错误报告
    logger.error("=" * 60)
    logger.error("JSON提取失败 - 所有方法都失败了")
    logger.error("=" * 60)
    
    # 🔍 详细诊断：检查原始文本特征
    logger.error("🔍 [诊断] 原始文本分析：")
    logger.error(f"   文本长度: {len(text)} 字符")
    logger.error(f"   是否包含 {{: {'是' if '{' in text else '否'}")
    logger.error(f"   是否包含 [: {'是' if '[' in text else '否'}")
    logger.error(f"   是否包含 ```json: {'是' if '```json' in text else '否'}")
    logger.error(f"   是否包含 ```: {'是' if '```' in text else '否'}")
    
    # 收集失败详情
    attempts_detail = []
    for i, candidate in enumerate(json_candidates[:3], 1):  # 只显示前3个
        preview = candidate[:100].replace('\n', ' ')
        attempts_detail.append(f"  候选 {i} (长度{len(candidate)}): {preview}...")
    
    error_msg = f"""无法从文本中提取有效的JSON

统计信息:
  - 尝试了 {len(json_candidates)} 个候选
  - 应用了 8 种修复策略（转义字符、引号、键、布尔值、注释等）"""
    
    if use_llm:
        error_msg += f"\n  - 包括 {max_attempts} 次LLM修复尝试"
    
    if attempts_detail:
        error_msg += f"\n\n候选详情（前3个）:\n" + "\n".join(attempts_detail)
    
    error_msg += f"\n\n原始文本长度: {len(text)} 字符"
    error_msg += f"\n原始文本预览:\n{text[:300]}"
    
    logger.error(error_msg)
    
    # 为调试提供完整信息
    logger.error("=" * 60)
    logger.error("🔍 [诊断] 完整原始文本：")
    logger.error(text)
    logger.error("=" * 60)
    
    raise json.JSONDecodeError(
        f"无法提取有效JSON (尝试了{len(json_candidates)}个候选)",
        text,
        0
    )
