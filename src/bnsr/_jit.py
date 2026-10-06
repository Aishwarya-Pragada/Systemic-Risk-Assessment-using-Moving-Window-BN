"""numba shim: use numba when installed, otherwise run the same code as plain Python.

The plain-Python fallback is only meant for the unit tests on tiny problems; real runs need numba.
"""
try:
    from numba import njit  # noqa: F401
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(f):
            return f

        return deco
