class SizeMismatchError(Exception):
    """ Raised when the size of the objects does not match
    """
    def __init__(self, msg):
        self.msg = msg
