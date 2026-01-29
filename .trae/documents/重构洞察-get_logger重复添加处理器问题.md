# 1. 问题

当前统一日志工具函数 `get_logger` 在每次调用时都会为同名 logger 新增 `StreamHandler` 与可选的 `FileHandler`，而没有检查是否已存在相同类型或相同路径的处理器。这会导致同一条日志被重复打印，且重复次数随着调用次数增长。问题集中在 `llm_data_wash/utils/logger.py` 第 11-27 行；受影响的调用点包括 `generators/vllm_generator.py`、`utils/gpu_monitor.py`、`utils/concurrency.py` 与 `scripts/run_pipeline.py`。

## 1.1. **处理器重复添加**

- 位置：`llm_data_wash/utils/logger.py` 11-27 行。
- 现状：每次调用都会执行 `logger.addHandler(console_handler)`；当传入 `log_file` 时也会重复添加 `FileHandler`。
- 影响：
  - 同一条日志被打印多次，排障噪声增加。
  - I/O 与格式化开销按调用次数累积，长跑任务与高并发下放大。
  - 线上日志体量不稳定，影响告警规则与容量规划。
- 代表性代码（现状）：

```python
# llm_data_wash/utils/logger.py
console_handler = logging.StreamHandler()
console_handler.setLevel(level)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

if log_file:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
```

## 1.2. **并发下的竞态问题**

- 位置：同上。
- 现状：若在多线程场景同时调用 `get_logger`，即使加入了“是否存在处理器”的判断，也可能在检查与添加之间产生竞态，从而仍旧重复添加。
- 影响：并发下重复率更高，日志写放大更明显，且问题更随机难复现。

# 2. 收益

一句话总结：将 `get_logger` 改造成“幂等 + 线程安全”的处理器管理，保证同名 logger 仅绑定一套处理器，在不改变外部调用方式的前提下降低日志噪声与 I/O 开销。

## 2.1. **重复写入明显下降**

- 同一日志由多次输出收敛为一次。以三个模块各自调用一次为例，重复输出由 **3** 次降为 **1** 次；累计 I/O 与格式化开销同步下降。

## 2.2. **日志稳定性与可读性提升**

- 输出量与格式稳定，排障时更易定位根因；监控与告警阈值更加可控。

## 2.3. **可维护性与扩展性更好**

- 统一在工具函数内做去重与更新，后续调整级别、格式或切换文件路径时无需改动调用点；行为可预测。

## 2.4. **并发安全**

- 通过锁保护“检查—添加—更新”的临界区，避免并发下重复添加处理器。

# 3. 方案

总体思路：让 `get_logger` 按处理器类型与文件路径进行“存在性检查 + 条件更新”，并在并发场景下通过锁保证幂等。

## 3.1. **解决“处理器重复添加”与“并发竞态”**

- 方案概述：
  - 对控制台处理器：仅当不存在非文件的 `StreamHandler` 时添加；若已存在，仅更新其 `level` 与 `formatter`。
  - 对文件处理器：以绝对路径为唯一标识；不存在则添加；存在但路径变化则替换；路径一致则仅更新 `level` 与 `formatter`。
  - 使用模块级 `Lock` 保护检查到添加/替换的临界区，避免并发竞态。

- 实施步骤：
  - 引入 `threading.Lock`，在 `get_logger` 内部用 `with lock:` 包裹处理器检查与变更。
  - 使用 `isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)` 区分控制台处理器，避免把 `FileHandler` 误判为控制台（`FileHandler` 继承自 `StreamHandler`）。
  - 文件路径统一用 `os.path.abspath` 比较；必要时安全关闭旧的 `FileHandler` 后替换为新路径。
  - 始终将已有处理器的 `level` 与 `formatter` 更新为最新调用参数，保证行为一致性。

- 修改前代码（节选）：

```python
# 每次都会新增处理器
logger.addHandler(console_handler)
...
logger.addHandler(file_handler)
```

- 修改后代码（示例实现）：

```python
# llm_data_wash/utils/logger.py
import logging
import os
import threading
from typing import Optional

_lock = threading.Lock()

def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    with _lock:
        # 控制台处理器（排除 FileHandler，因为其继承自 StreamHandler）
        console_handlers = [h for h in logger.handlers
                            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        if console_handlers:
            ch = console_handlers[0]
            ch.setLevel(level)
            ch.setFormatter(formatter)
        else:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        # 文件处理器（按绝对路径去重/替换/更新）
        if log_file:
            abs_path = os.path.abspath(log_file)
            dir_name = os.path.dirname(abs_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            fh_existing = None
            for h in logger.handlers:
                if isinstance(h, logging.FileHandler):
                    fh_existing = h
                    break

            if fh_existing is None:
                fh = logging.FileHandler(abs_path, encoding="utf-8")
                fh.setLevel(level)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            else:
                # 路径变更则替换；路径一致则更新
                if getattr(fh_existing, "baseFilename", None) != abs_path:
                    logger.removeHandler(fh_existing)
                    try:
                        fh_existing.close()
                    except Exception:
                        pass
                    fh = logging.FileHandler(abs_path, encoding="utf-8")
                    fh.setLevel(level)
                    fh.setFormatter(formatter)
                    logger.addHandler(fh)
                else:
                    fh_existing.setLevel(level)
                    fh_existing.setFormatter(formatter)

    return logger
```

- 说明：
  - 保持对外不变的调用方式（`get_logger(__name__, log_file=..., level=...)`）。
  - 当后续以不同 `level` 或新的 `log_file` 再次调用时，现有处理器会被更新或替换，不会产生重复输出。

# 4. 回归范围

从“端到端业务过程”角度覆盖可能受影响的日志行为，重点验证“无重复输出、级别与格式一致、文件路径切换正确”。

## 4.1. 主链路

- 运行 `scripts/run_pipeline.py`：
  - 前置：确保同一进程内多处调用 `get_logger(__name__, log_file="./regenerate.log")`。
  - 步骤：运行一次完整管道；观察控制台与 `./regenerate.log`。
  - 期望：每条日志仅输出一次；文件与控制台内容一致，格式稳定。

- `generators/vllm_generator.py` 批处理日志：
  - 前置：并发线程池处理数据，包含大量 `logger.info`。
  - 期望：并发下仍保持单次输出，无随机重复或丢失。

- `utils/gpu_monitor.py` 周期性日志：
  - 前置：启动与停止监控，间隔打印状态。
  - 期望：周期性输出无重复，停止后不再输出；级别与格式正确。

## 4.2. 边界情况

- 二次调用更新级别：先以 `DEBUG` 获取，再以 `INFO` 获取。
  - 期望：不新增处理器；现有处理器级别更新为最新。

- 从无文件到有文件：先 `log_file=None` 调用，随后以路径调用。
  - 期望：仅添加一个文件处理器；控制台仍唯一。

- 文件路径切换：同名 logger 先写 `a.log`，后切到 `b.log`。
  - 期望：旧处理器被安全替换；新文件产生日志，仅一次输出。

- 并发调用：多线程同时调用 `get_logger`。
  - 期望：无重复添加；输出稳定。
