from __future__ import print_function
DEBUG = True


def log_print(msg, end_=None):
    if DEBUG is True and end_ is not None:
        print(msg, end=end_)
    elif DEBUG is True:
        print(msg)
