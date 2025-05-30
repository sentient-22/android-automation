import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output"""
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    blue = "\x1b[34;20m"
    
    FORMATS = {
        logging.DEBUG: f"{grey}%(levelname)s{reset}: %(message)s",
        logging.INFO: f"%(message)s",
        logging.WARNING: f"{yellow}%(levelname)s{reset}: %(message)s",
        logging.ERROR: f"{red}%(levelname)s{reset}: %(message)s",
        logging.CRITICAL: f"{bold_red}%(levelname)s{reset}: %(message)s"
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    mode: str = 'a',
    console: bool = True
) -> logging.Logger:
    """Setup a logger with both console and file handlers.
    
    Args:
        name: Logger name
        log_level: Logging level (default: logging.INFO)
        log_file: Path to log file (optional)
        mode: File mode ('w' to overwrite, 'a' to append)
        console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicate logs
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode=mode, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)
    
    return logger

def get_timestamp() -> str:
    """Get current timestamp in a consistent format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a default logger instance
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"app_{get_timestamp()}.log"
logger = setup_logger(__name__, log_file=log_file)

# Example usage
if __name__ == "__main__":
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
