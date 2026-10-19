"""
轻量日志工具，避免在各脚本里重复配置 logging。
"""

import logging
from logging import Logger
from typing import Optional

__all__ = ["get_logger", "setup_logging"]


def setup_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    设置全局基础日志配置，只在未配置时生效。
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    else:
        logging.getLogger().setLevel(level)


def get_logger(
    name: str = "hydro",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> Logger:
    """
    获取带基本格式的 logger，可选输出到文件。
    """
    setup_logging(level=level)
    logger = logging.getLogger(name)

    # 已存在同类 handler 则不重复添加
    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(stream_handler)

    if log_file:
        has_file = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == log_file
            for h in logger.handlers
        )
        if not has_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    "%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger