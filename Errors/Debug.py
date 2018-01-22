from __future__ import print_function
DEBUG = False


def log_print(msg, end_=None):
    """ Only prints the given message if the DEBUG mode is True

    Paramters
    ---------
    msg : str
        The message
    end_ : str
        This will be passed to print as its 'end' argument
    """
    if DEBUG is True and end_ is not None:
        print(msg, end=end_)
    elif DEBUG is True:
        print(msg)
