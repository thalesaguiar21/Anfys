from Errors import SizeMismatchError
from exceptions import AttributeError


class Alphabet():
    """
    """
    def __init__(self, symbols, values):
        self._symbols = symbols
        self._values = values
        self.validate()

    def __getitem__(self, symb):
        idx = self._symbols.index(symb)
        return self._values[idx]

    def validate(self):
        if self._symbols is None:
            raise AttributeError
        if self._values is None:
            raise AttributeError

        numOfSymbs = len(self._symbols)
        if numOfSymbs != len(self._values):
            raise SizeMismatchError()

    def get_pair(self, index):
        pair = None
        if index >= 0 and index < len(self._symbols):
            pair = (self._symbols[index], self._values[index])
        return pair
