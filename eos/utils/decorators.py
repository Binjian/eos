import functools

'''
function decorator to prepend a string argument to a function
'''
def prepend_string_arg(strArg="TQD_trqTrqSetNormal_MAP_v"):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func.__name__
            return func(strArg, *args, **kwargs)
        return wrapper
    return decorate
