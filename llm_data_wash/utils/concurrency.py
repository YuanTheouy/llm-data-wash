from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Tuple
from tqdm import tqdm
from llm_data_wash.utils.logger import get_logger

logger = get_logger(__name__)

def run_thread_pool(
    func: Callable,
    tasks: List[Tuple],
    max_workers: int = 32
) -> List:
    """
    通用线程池执行函数
    :param func: 要执行的函数
    :param tasks: 任务列表，每个元素是func的参数元组
    :param max_workers: 最大线程数
    :return: 执行结果列表
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(func, *task): task for task in tasks}
        
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="处理进度"):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"处理任务 {task} 失败: {e}")
                results.append(None)
    
    return results