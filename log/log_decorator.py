# coding=utf8
import os
import sys
import time
import logging

# 详细的 logging模块可参考：https://cuiqingcai.com/6080.html
# 带参数的装饰器需要2层装饰器实现，第一层传参数，第二层传函数，每层函数在上一层返回，可选是否记录到文件
def log(filename="./out.log", to_file=False):
    def decorator(func):
        def inner(*args, **kwargs):
            logger = logging.getLogger()
            logger.setLevel(level=logging.DEBUG)

            # StreamHandler
            stream_handler = logging.StreamHandler(sys.stdout)      # 输出到控制台
            stream_handler.setLevel(level=logging.INFO)
            formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                              fmt='%(asctime)s - {} function - %(message)s'.format(func.__name__))
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

            # FileHandler
            if to_file:
                filepath = os.path.dirname(filename)
                if not os.path.exists(filepath): os.makedirs(filepath)

                file_handler = logging.FileHandler(filename)        # 输出到文件
                file_handler.setLevel(level=logging.WARNING)
                formatter = logging.Formatter('%(asctime)s - {} function - %(levelname)s - %(message)s'.format(func.__name__))
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            try:
                logger.info("Start running.")
                init_time = time.time()
                res = func(*args, **kwargs)
                end_time = time.time()
                logger.info("Finish running. The running time is {:.4} minutes.".format((end_time-init_time)/60))
                return res
            except Exception:
                logger.error("Run Error", exc_info=True)
        return inner
    return decorator


def log_not_to_file(func):
    def inner(*args, **kwargs):
        logger = logging.getLogger()
        logger.setLevel(level=logging.DEBUG)

        # StreamHandler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='%(asctime)s - {} function - %(message)s'.format(func.__name__))
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        try:
            logger.info("Start running.")
            init_time = time.time()
            res = func(*args, **kwargs)
            end_time = time.time()
            logger.info("Finish running. The running time is {:.4} minutes.".format((end_time-init_time)/60))
            return res
        except Exception:
            logger.error("Run Error", exc_info=True)
    return inner
