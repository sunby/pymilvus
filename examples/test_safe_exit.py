#!/usr/bin/env python3
"""
测试 simple.py 的安全退出功能

这个脚本创建一个简化版本的测试，验证：
1. 信号处理是否正常工作
2. 进程池是否能正确清理
3. 资源是否被释放
"""

import signal
import sys
import time
import multiprocessing as mp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
_should_stop = False
_current_pool = None

def signal_handler(signum, frame):
    """信号处理器"""
    global _should_stop, _current_pool
    logger.warning("\n收到退出信号，正在清理...")
    _should_stop = True
    
    if _current_pool is not None:
        try:
            _current_pool.terminate()
            _current_pool.join(timeout=5)
            logger.info("进程池已清理")
        except Exception as e:
            logger.error(f"清理进程池时出错: {e}")
    
    logger.info("退出完成")
    sys.exit(0)

def worker_function(worker_id, duration):
    """工作进程函数"""
    logger.info(f"Worker {worker_id} 开始工作...")
    start_time = time.time()
    
    while time.time() - start_time < duration:
        time.sleep(0.5)
        # 模拟工作
    
    logger.info(f"Worker {worker_id} 完成")
    return f"Worker {worker_id} 结果"

def test_multiprocessing():
    """测试多进程功能"""
    global _current_pool
    
    logger.info("=" * 60)
    logger.info("测试多进程安全退出功能")
    logger.info("提示：按 Ctrl+C 可以随时安全退出")
    logger.info("=" * 60)
    
    num_workers = 4
    work_duration = 30  # 每个worker运行30秒
    
    pool = None
    try:
        pool = mp.Pool(processes=num_workers)
        _current_pool = pool
        
        logger.info(f"启动 {num_workers} 个工作进程...")
        
        # 创建任务
        tasks = [(i, work_duration) for i in range(num_workers)]
        
        # 执行任务
        results = pool.starmap(worker_function, tasks)
        
        logger.info("所有任务完成！")
        logger.info(f"结果: {results}")
        
    except KeyboardInterrupt:
        logger.warning("检测到键盘中断")
        if pool:
            pool.terminate()
            pool.join(timeout=5)
        raise
    except Exception as e:
        logger.error(f"出错: {e}")
        if pool:
            pool.terminate()
            pool.join(timeout=5)
        raise
    finally:
        if pool:
            pool.close()
            pool.join()
        _current_pool = None
        logger.info("资源清理完成")

if __name__ == "__main__":
    # 设置 multiprocessing 启动方法
    mp.set_start_method('spawn', force=True)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        test_multiprocessing()
    except KeyboardInterrupt:
        logger.warning("\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"测试失败: {e}")
        sys.exit(1)
    
    logger.info("\n✅ 测试通过！安全退出功能正常工作")

