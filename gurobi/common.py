import time
import logging
import sys


log = logging.getLogger('control')
log.setLevel(logging.NOTSET+1)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.WARN)
# console.setLevel(logging.INFO)
log.addHandler(console)


def memoize(func):
    cache = dict()

    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func


def log_time(func=None, logger=log.debug):

    def decorator(func):
        def wrapped_func(*args, **kwargs):
            timer_start = time.time()
            results = func(*args, **kwargs)
            timer_end = time.time()
            logger("Function: {} took {} seconds"
                   .format(func.__qualname__, timer_end - timer_start))
            return results
        return wrapped_func

    if(func is not None):
        return decorator(func)
    return decorator


class Namespace:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    # def __getattr__(self, attr):
    #     if attr in self.__dict__:
    #         return self.__dict__[attr]
    #     else:
    #         raise AttributeError("Not found key: {}".format(attr))

    def frozen_setter(self, attr):
        raise AttributeError("Object {} is frozen,"
                             " can't set attribute:"
                             .format(self, attr))

    def __repr__(self):
        return repr(self.__dict__)


def freeze_object(obj):
    if(isinstance(obj, Namespace)):
        obj.__setter__ = obj.frozen_setter
    else:
        raise TypeError("obj: {} is not a Namespace instance", obj)


# class InfoFilter(logging.Filter):
#     def filter(self, rec):
#         return rec.levelno == logging.INFO


# class DebugFilter(logging.Filter):
#     def filter(self, rec):
#         return rec.levelno == logging.DEBUG


def remove_all_file_loggers():
    to_remove = []
    for handler in log.handlers:
        if(isinstance(handler, logging.FileHandler)):
            to_remove.append(handler)

    for h in to_remove:
        log.removeHandler(h)


def add_file_logger(file_path):
    file_handler = logging.FileHandler(file_path, mode='w')
    formatter = logging.Formatter(
        '%(asctime)s:%(levelname)s::%(message)s')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)


def setup_logging(args):

    if(args.verbose >= 4):
        console.setLevel(logging.DEBUG-2)
    elif(args.verbose >= 3):
        console.setLevel(logging.DEBUG-1)
    elif(args.verbose >= 2):
        console.setLevel(logging.DEBUG)
    elif(args.verbose >= 1):
        console.setLevel(logging.INFO)
    else:
        console.setLevel(logging.WARNING)

    if(args.output_file):
        add_file_logger(args.output_file)

    # if(args.verbose >= 2):
    #     h_debug = logging.StreamHandler(sys.stdout)
    #     h_debug.setLevel(logging.DEBUG)
    #     h_debug.addFilter(DebugFilter())
    #     log.addHandler(h_debug)
    #     # TODO: Add option to redirect debug to file
    # if(args.verbose >= 1):
    #     h_info = logging.StreamHandler(sys.stdout)
    #     h_info.setLevel(logging.INFO)
    #     h_info.addFilter(InfoFilter())
    #     log.addHandler(h_info)
    # if(args.verbose >= 0):
    #     h_warn = logging.StreamHandler() # stderr default
    #     h_warn.setLevel(logging.WARNING)
    #     log.addHandler(h_warn)


constants = Namespace(
    cell_size=4,
    KB2B=1024,
    NS_LARGEST=1000
)
