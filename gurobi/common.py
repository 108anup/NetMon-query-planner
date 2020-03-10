import time
import logging
import sys


def memoize(func):
    cache = dict()

    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func


def log_time(func):

    def wrapped_func(*args, **kwargs):
        timer_start = time.time()
        results = func(*args, **kwargs)
        timer_end = time.time()
        log.info("Function: {} took {} seconds"
                 .format(func.__name__, timer_end - timer_start))
        return results
    return wrapped_func


class Namespace:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            raise AttributeError("Not found key: {}".format(attr))


class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno == logging.INFO


class DebugFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno == logging.DEBUG


def setup_logging(args):
    if(args.verbose >= 2):
        log.setLevel(logging.DEBUG)
    elif(args.verbose >= 1):
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.WARNING)

    if(args.verbose >= 2):
        h_debug = logging.StreamHandler(sys.stdout)
        h_debug.setLevel(logging.DEBUG)
        h_debug.addFilter(DebugFilter())
        log.addHandler(h_debug)
        # TODO: Add option to redirect debug to file
    if(args.verbose >= 1):
        h_info = logging.StreamHandler(sys.stdout)
        h_info.setLevel(logging.INFO)
        h_info.addFilter(InfoFilter())
        log.addHandler(h_info)
    if(args.verbose >= 0):
        h_warn = logging.StreamHandler()
        h_warn.setLevel(logging.WARNING)
        log.addHandler(h_warn)


log = logging.getLogger('control')

constants = Namespace(
    cell_size=4,
    KB2B=1024
)
