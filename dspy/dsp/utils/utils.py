import copy
import datetime


def print_message(*s, condition=True, pad=False, sep=None):
    """Print a timestamped message to stdout.
    
    Args:
        *s: Values to print.
        condition: If False, don't print (default: True).
        pad: If True, add newlines around the message (default: False).
        sep: Separator between values (default: None, uses space).
    
    Returns:
        The printed message string.
    """
    s = " ".join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        msg = msg if not pad else f"\n{msg}\n"
        print(msg, flush=True, sep=sep)

    return msg


def deduplicate(seq: list[str]) -> list[str]:
    """
        From Raymond Hettinger
        https://twitter.com/raymondh/status/944125570534621185
        Since Python 3.6 Dict are ordered
        Benchmark: https://gist.github.com/peterbe/67b9e40af60a1d5bcb1cfb4b2937b088
    """
    return list(dict.fromkeys(seq))


class dotdict(dict):  # noqa: N801
    def __getattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            return super().__getattr__(key)
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key.startswith("__") and key.endswith("__"):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __delattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            super().__delattr__(key)
        else:
            del self[key]

    def __deepcopy__(self, memo):
        # Use the default dict copying method to avoid infinite recursion.
        return dotdict(copy.deepcopy(dict(self), memo))
