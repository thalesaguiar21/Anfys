from __future__ import print_function


class Debugger():
    """
    """
    def __init__(self, debug):
        self.debug = debug

    def print(self, msg, end_line=True):
        if self.debug:
            if end_line:
                print(msg)
            else:
                print(msg, end="")
