import importlib
from typing import Any, Dict, Callable, Union, Optional

def str_to_obj(obj_path: Optional[Union[str, Dict, Callable, Any]]) -> Union[Dict, Callable, Any]:
    """
    将字符串路径转换为实际对象（可调用对象或字典）
    
    Args:
        obj_path: 可以是以下几种类型：
                 - 字符串：模块路径，如 "module.submodule.function_or_dict"
                 - 字典：直接返回
                 - 可调用对象：直接返回
                 - None：返回None
                 - 其它类型：直接返回原对象
    
    Returns:
        Union[Dict, Callable, Any]: 转换后的对象
    
    Examples:
        >>> str_to_obj("module.submodule.my_function")  # 返回my_function函数对象
        >>> str_to_obj("module.submodule.my_dict")      # 返回my_dict字典
        >>> str_to_obj({"a": 1})                        # 直接返回输入的字典
        >>> str_to_obj(lambda x: x+1)                   # 直接返回输入的函数
    """
    # 如果输入为None，直接返回None
    if obj_path is None:
        return None
    
    # 如果不是字符串，直接返回原对象
    if not isinstance(obj_path, str):
        return obj_path
    
    try:
        # 分割模块路径和对象名
        if '.' in obj_path:
            module_path, obj_name = obj_path.rsplit('.', 1)
            # 导入模块
            module = importlib.import_module(module_path)
            # 获取对象
            return getattr(module, obj_name)
        else:
            # 如果没有点号，视为单独的模块名或全局变量名
            return importlib.import_module(obj_path)
    except (ImportError, AttributeError) as e:
        # 导入失败，返回原字符串
        print(f"无法导入对象 {obj_path}: {str(e)}")
        return obj_path