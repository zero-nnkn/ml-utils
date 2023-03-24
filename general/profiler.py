import contextlib
import time

import torch


def sync_time():
    """
    It synchronizes the time on the GPU with the time on the CPU
    
    Returns:
      The time in seconds since the epoch as a floating point number.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 Profile class.
    Usage: as a decorator with @Profile() or as a context manager with 'with Profile():'
    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/utils/ops.py
    """

    def __init__(self, t=0.0):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
        """
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        """
        Start timing.
        """
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """
        Stop timing.
        """
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """
        Get current time.
        """
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()