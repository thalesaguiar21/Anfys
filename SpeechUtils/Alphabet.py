from Errors import SizeMismatchError
from exceptions import AttributeError


class Alphabet():
    """
    """
    def __init__(self, symbols, values):
        self.symbols = symbols
        self.values = values
        self.validate()

    def validate(self):
        if self.symbols is None:
            raise AttributeError
        if self.values is None:
            raise AttributeError

        numOfSymbs = len(self.symbols)
        if numOfSymbs != len(self.values):
            raise SizeMismatchError()

    def get_pair(self, index):
        pair = None
        if index >= 0 and index < len(self.symbols):
            pair = (self.symbols[index], self.values[index])
        return pair
