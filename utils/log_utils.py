import logging
import os


# 配置日志记录器
def configure_logging(model_name):
    # 创建 logger
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)  # 设置日志记录级别

    # 检查是否已经有处理程序，避免重复添加
    if not logger.hasHandlers():
        # 创建 formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 创建 console handler 并设置级别为 info
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)  # 添加 formatter 到 ch
        logger.addHandler(ch)  # 将 ch 添加到 logger

        logs_dir = os.path.join("./log", model_name)
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # 创建 file handler，写入日志文件，设置级别为 info
        fh = logging.FileHandler(os.path.join(logs_dir, "ModelTraining.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)  # 添加 formatter 到 fh
        logger.addHandler(fh)  # 将 fh 添加到 logger

    return logger
