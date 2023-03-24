import functools

"""
function decorator to prepend a string argument to a function
"""


def prepend_string_arg(strArg='TQD_trqTrqSetNormal_MAP_v'):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            __name__ = func.__name__
            __doc__ = func.__doc__
            return func(strArg, *args, **kwargs)

        return wrapper

    return decorate
