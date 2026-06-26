import threading
import time

def rate_limit(calls_per_minute, time_window=60):
  """
  一个线程安全的 RateLimit 装饰器，用于控制在指定时间窗口内 API 调用的数量。

  Args:
    calls_per_minute: 每分钟允许的最大调用次数。
    time_window: 时间窗口（以秒为单位）。默认为 60 秒。

  Returns:
    一个装饰器函数。
  """
  lock = threading.Lock()
  call_count = 0
  last_reset_time = time.time()

  def decorator(func):
    def wrapper(*args, **kwargs):
      nonlocal call_count, last_reset_time
      with lock:
        if time.time() - last_reset_time > time_window:
          call_count = 0
          last_reset_time = time.time()

        if call_count < calls_per_minute:
          call_count += 1
          return func(*args, **kwargs)
        else:
          raise Exception("API 调用频率超过限制")

    return wrapper

  return decorator