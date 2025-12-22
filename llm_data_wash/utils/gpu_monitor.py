import pynvml
import threading
import time
from llm_data_wash.utils.logger import get_logger

logger = get_logger(__name__)

class GPUMonitor:
    """8卡GPU监控工具"""
    
    def __init__(self, gpu_count: int = 8):
        self.gpu_count = gpu_count
        pynvml.nvmlInit()
        self._monitor_thread = None
        self._stop_flag = False
    
    def start_monitor(self, interval: int = 10):
        """启动GPU监控后台线程"""
        self._stop_flag = False
        self._monitor_thread = threading.Thread(
            target=self._monitor_worker,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("GPU监控线程已启动")
    
    def stop_monitor(self):
        """停止监控线程"""
        self._stop_flag = True
        if self._monitor_thread:
            self._monitor_thread.join()
        pynvml.nvmlShutdown()
        logger.info("GPU监控线程已停止")
    
    def _monitor_worker(self, interval: int):
        """监控工作线程"""
        while not self._stop_flag:
            self._log_gpu_status()
            time.sleep(interval)
    
    def _log_gpu_status(self):
        """打印GPU状态"""
        total_util = 0
        logger.info("\n=== GPU状态监控 ===")
        for gpu_idx in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                mem_used = mem_info.used / (1024 ** 3)
                mem_total = mem_info.total / (1024 ** 3)
                gpu_util = util_rates.gpu
                
                total_util += gpu_util
                logger.info(
                    f"GPU {gpu_idx}: 显存 {mem_used:.1f}GB/{mem_total:.1f}GB | 利用率 {gpu_util:.1f}%"
                )
            except Exception as e:
                logger.error(f"GPU {gpu_idx} 监控失败: {e}")
        logger.info(f"平均GPU利用率: {total_util/self.gpu_count:.1f}%\n")