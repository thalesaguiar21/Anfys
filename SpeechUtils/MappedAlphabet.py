from Errors import SizeMismatchError


class MappedAlphabet:
    """ This class represents a data structure that simulates
    an alphabet where each symbol has a value mapped to it.random
    Note that this differs from a dict, since the elemets order
    are important.
    """

    def __init__(self, symbols, thresholds):
        """ Create a new MappedAlphabet that assign a value to
        each symbol

        Parameters
        ----------
        symbols : list
            The alphabet's symbols
        thresholds: list with the partially ordered property
            A list with each value of the alphabet symbol
        """
        if(len(symbols) != len(thresholds)):
            raise SizeMismatchError('Number of symbols and values differ!')
        self.__symbols = symbols
        self.__thresholds = thresholds
        self.__size = len(symbols)

    def add(self, symbol, value):
        """ Insert the given symbol and value in the alphabet, preserving the
        order.

        Parameters:
        symbol : Object
            The new alphabet's symbol
        value : Object
            The value associated with the symbol
        """
        for i in range(self.__size):
            if value < self.__thresholds[i]:
                self.__symbols.insert(i, symbol)
                self.__thresholds.insert(i, value)
                break
        else:
            self.__symbols.insert(i, symbol)
            self.__thresholds.insert(i, value)
        self.__size += 1

    def select(self, pos):
        """ Select the pair symbol -> value at a position

        Parameters
        ----------
        pos : int
            The position of the desired item

        Returns
        -------
        symbol : Object
            The symbol at the given position, or None if the position is not in
            range
        value : Object
            The value associated to the symbol in the given position, or None
            if the position is not in range.
        """
        symbol, value = None, None
        if pos < self.__size and pos > -1:
            symbol = self.__symbols[pos]
            value = self.__thresholds[pos]
        return symbol, value

    def remove(self, symbol):
        """ Remove the given symbol, and its repective value, from the alphabet

        Parameters
        ----------
        symbol : Object
            The symbol to be removed from the alphabet
        """
        idx = self.index(symbol)
        print 'Found symbol \'{}\' at position {}'.format(symbol, idx)
        if idx is not None:
            print 'Removing element at index ' + str(idx)
            del self.__symbols[idx]
            del self.__thresholds[idx]
            self.__size -= 1

    def index(self, symbol):
        """ The index of a given symbol

        Parameters
        ----------
        symbol : Object
            A symbol in the alphabet

        Returns
        -------
        index : int
            The index of the given symbol in the alphabet, or None if the
            symbols does not belong to it.
        """
        index = None
        if symbol in self.__symbols:
            index = self.__symbols.index(symbol)
        return index

    def value(self, symbol):
        """ The value assigned to the given alphabet's symbols

        Parameters
        ----------
        symbol : Object
            A aplhabet's symbol

        Returns
        -------
        value : Object
            The value assigned to this symbol, or None if the given symbol
            does not belong to the alphabet.
        """
        value = None
        if symbol in self.__symbols:
            value = self.__thresholds[self.index(symbol)]
        return value

    def symbol(self, pos):
        """ The symbol at the given position

        Parameters
        ----------
        pos : int
            The position of the symbol

        Returns
        -------
        symbol : Object

        """
        symb = None
        if pos < self.__size and pos >= 0:
            symb = self.__symbols[pos]
        return symb

    def step(self, value):
        """ The symbol which the given value is in range. Since the
        thresholds of the alphabet are a partially ordered set, then
        we can define that given three symbols u, v and w, where
        next(u) = v, and next(v) = w. Therefore,
        value(u) <= value(v) <= value(w)
        That is, if the given value respects
        value(u) <= value <= value(w), the the symbol returned is v.

        Parameters
        ----------
        value : comparable Object
            A value assigned to an symbol

        Returns
        -------
        symbol : Object
            The symbol for which the given value is in range, or None if
            the value is outside all ranges.

        """
        symb = None
        for i in range(self.__size):
            if value <= self.__thresholds[i]:
                symb = self.__symbols[i]
                break
        return symb
