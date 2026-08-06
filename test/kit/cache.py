import functools
import inspect


def memo(fn):
    """
    One fixture shared across exams, keyed on RESOLVED arguments rather than the keywords
    used. ``share=False`` forces a fresh call.
    """
    sig = inspect.signature(fn)
    store = {}

    @functools.wraps(fn)
    def wrap(*args, share=True, **kw):
        bound = sig.bind(*args, **kw)
        bound.apply_defaults()
        key = tuple(bound.arguments.values())
        if share and key in store:
            return store[key]

        out = fn(*bound.args, **bound.kwargs)
        if share:
            store[key] = out

        return out

    wrap.store = store

    return wrap
