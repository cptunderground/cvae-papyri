import datetime
import time
from util.base_logger import logger


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        total = time.time() - start
        logger.info(F"Total time used for {func.__name__} - {str(datetime.timedelta(seconds=total))}")
        return result

    return wrapper
