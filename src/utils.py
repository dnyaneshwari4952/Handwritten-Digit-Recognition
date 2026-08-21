"""
src/utils.py
Utilities for reproducibility, hardware device management, logging, and I/O.
"""

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seed across all libraries to ensure full determinism and reproducibility.

    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Configure deterministic ops if available
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        logging.getLogger("utils").debug("TensorFlow not found during set_seed; skipping TF seeding.")


def get_device_info() -> Dict[str, Any]:
    """
    Detect available computational hardware (CPU, GPU, MPS, etc.).

    Returns:
        Dictionary containing platform and device details.
    """
    info = {
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "device": "CPU",
        "gpu_count": 0,
        "gpu_details": [],
    }

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            info["device"] = "GPU"
            info["gpu_count"] = len(gpus)
            info["gpu_details"] = [gpu.name for gpu in gpus]
        else:
            info["device"] = "CPU"
    except Exception as exc:
        logging.getLogger("utils").debug(f"GPU detection error: {exc}")

    return info


def setup_logger(name: str = "mnist_app", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a standardized console logger.

    Args:
        name: Logger name identifier.
        level: Logging level (default INFO).

    Returns:
        Configured logging.Logger object.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | [%(levelname)s] | %(name)s : %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Safely serialize dictionary data to a JSON file.

    Args:
        data: Dictionary data to persist.
        filepath: Destination file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serializable objects (like numpy types or Path objects)
    def default_converter(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, Path):
            return str(o)
        return str(o)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=default_converter)


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Safely load a JSON file into a dictionary.

    Args:
        filepath: Source file path.

    Returns:
        Dictionary loaded from JSON.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found at: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def timeit(description: str = "Operation"):
    """
    Context manager to benchmark and log the execution time of code blocks.

    Args:
        description: Name or description of the operation being timed.
    """
    logger = setup_logger("timer")
    start = time.perf_counter()
    logger.info(f"Starting {description}...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"Finished {description} in {elapsed:.3f} seconds.")
