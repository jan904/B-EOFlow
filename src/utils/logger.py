import logging
from datetime import datetime
import os


def get_logger(log_dir="logs", level=logging.INFO, console=False, log_file=None):
    """
    Create a standard logger that writes to a timestamped file and also prints to console.

    Parameters
    ----------
    log_dir : str
        Directory where log files will be saved.
    level : int
        Logging level (default: INFO)

    Returns
    -------
    logger : logging.Logger
        Configured logger instance
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    resume = True if log_file is not None else False

    # Timestamp for filename
    if log_file is not None:
        log_path = os.path.join(log_dir, log_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"log_{timestamp}.log")

    file_mode = "a" if resume else "w"

    # Create logger
    logger_name = f"Logger_{os.path.basename(log_path)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, mode=file_mode)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # Console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(ch)

    return logger, log_path
