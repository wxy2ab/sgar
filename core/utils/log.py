import logging
import os
from datetime import datetime

try:
    from colorlog import ColoredFormatter
except ImportError:
    ColoredFormatter = None


def _build_console_formatter(log_colors=None, format_str=None):
    if ColoredFormatter is not None:
        return ColoredFormatter(
            format_str or "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors=log_colors
            or {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

    return logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def setup_logger():
    """
    设置日志记录器，同时输出到控制台和文件。

    Returns:
        logging.Logger: 配置好的日志记录器。
    """
    log_level = os.environ.get('LOG_LEVEL', 'ERROR')
    logger = logging.getLogger("logger")
    level = getattr(logging, log_level.upper())
    logger.setLevel(level)
    
    # 防止日志传播到父logger，避免重复输出
    logger.propagate = False
    
    # 如果已经有handler了，说明已经初始化过，直接返回
    if logger.handlers:
        return logger

    # 创建一个控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 创建一个彩色格式化器（只在控制台输出）
    color_formatter = _build_console_formatter()

    # 设置控制台处理器的格式化器
    console_handler.setFormatter(color_formatter)

    # 将控制台处理器添加到 logger
    logger.addHandler(console_handler)

    # 创建日志文件目录
    log_dir = "./output/log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建文件处理器，每天生成一个新的日志文件
    log_filename = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(level)  # 文件处理器的日志级别与logger保持一致

    # 创建文件格式化器（不输出彩色）
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 设置文件处理器的格式化器
    file_handler.setFormatter(file_formatter)

    # 将文件处理器添加到 logger
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "logger"):
    """
    获取指定名称的日志记录器（为每个模块提供独立的 logger）。
    
    Args:
        name: logger 的名称，通常使用 __name__ 传入
        
    Returns:
        logging.Logger: 配置好的日志记录器
        
    示例:
        from core.utils.log import get_logger
        logger = get_logger(__name__)
        logger.info("这是一条日志")
    """
    log_level = os.environ.get('LOG_LEVEL', 'ERROR')
    logger_instance = logging.getLogger(name)
    
    # 防止日志传播到父logger，避免重复输出
    logger_instance.propagate = False
    
    # 如果 logger 已经有 handler，说明已经配置过，直接返回
    if logger_instance.handlers:
        return logger_instance
    
    level = getattr(logging, log_level.upper())
    logger_instance.setLevel(level)

    # 创建一个控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 创建一个彩色格式化器（只在控制台输出）
    color_formatter = _build_console_formatter()

    # 设置控制台处理器的格式化器
    console_handler.setFormatter(color_formatter)

    # 将控制台处理器添加到 logger
    logger_instance.addHandler(console_handler)

    # 创建日志文件目录
    log_dir = "./output/log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建文件处理器，每天生成一个新的日志文件
    log_filename = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(level)  # 文件处理器的日志级别与logger保持一致

    # 创建文件格式化器（不输出彩色）
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 设置文件处理器的格式化器
    file_handler.setFormatter(file_formatter)

    # 将文件处理器添加到 logger
    logger_instance.addHandler(file_handler)

    return logger_instance


def get_agent_logger(agent_name: str, is_planner: bool = False):
    """
    获取带Agent名称标注的日志记录器
    
    Args:
        agent_name: Agent的名称
        is_planner: 是否是负责思考调度的Agent（Planner/Orchestrator）
        
    Returns:
        logging.Logger: 配置好的日志记录器
        
    示例:
        from core.utils.log import get_agent_logger
        logger = get_agent_logger("backtest_agent")
        logger.info("执行回测任务")
    """
    log_level = os.environ.get('LOG_LEVEL', 'ERROR')
    logger_name = f"Agent.{agent_name}"
    logger_instance = logging.getLogger(logger_name)
    
    # 防止日志传播到父logger，避免重复输出
    logger_instance.propagate = False
    
    # 如果 logger 已经有 handler，说明已经配置过，直接返回
    if logger_instance.handlers:
        return logger_instance
    
    level = getattr(logging, log_level.upper())
    logger_instance.setLevel(level)

    # 创建一个控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 根据是否是Planner使用不同的颜色方案
    if is_planner:
        # Planner使用更醒目的颜色：粗体+亮色
        log_colors = {
            "DEBUG": "bold_cyan",
            "INFO": "bold_white",
            "WARNING": "bold_yellow",
            "ERROR": "bold_red",
            "CRITICAL": "bold_red,bg_white",
        }
        # 添加特殊标记
        format_str = "%(log_color)s🧠 %(asctime)s - [%(name)s] - %(levelname)s - %(message)s%(reset)s"
    else:
        # 普通Agent使用常规颜色
        log_colors = {
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        }
        format_str = "%(log_color)s%(asctime)s - [%(name)s] - %(levelname)s - %(message)s%(reset)s"

    # 创建一个彩色格式化器
    color_formatter = _build_console_formatter(
        log_colors=log_colors,
        format_str=format_str,
    )

    # 设置控制台处理器的格式化器
    console_handler.setFormatter(color_formatter)

    # 将控制台处理器添加到 logger
    logger_instance.addHandler(console_handler)

    # 创建日志文件目录
    log_dir = "./output/log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建文件处理器，每天生成一个新的日志文件
    log_filename = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(level)  # 文件处理器的日志级别与logger保持一致

    # 创建文件格式化器（不输出彩色）
    planner_prefix = "🧠 " if is_planner else ""
    file_formatter = logging.Formatter(
        f"{planner_prefix}%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 设置文件处理器的格式化器
    file_handler.setFormatter(file_formatter)

    # 将文件处理器添加到 logger
    logger_instance.addHandler(file_handler)

    return logger_instance


# 使用示例
logger = setup_logger()
