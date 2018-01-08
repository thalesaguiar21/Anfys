phonemeList = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER',
    'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z',
    'ZH'
]

phonemes = {}


def phoneme(i):
    """ The i-th phoneme from the current dictionary

    Parameters
    ----------
    i : int
        The position of the phoneme, starting from 0.

    Returns
    -------
    phone : str
        An string with the i-th phoneme.

    Raise:
    -----
    IndexOutOfRange -- If the value passed is not within the dictionary range.
    """
    return phonemeList[i]


def index(chars):
    """ The index of an given phoneme in the Dictionary.

    Parameters
    ----------
    chars : str
        The phoneme

    Returns
    -------
    index : int
        The index of the phoneme in the dictionary

    Raise
    -----
    KeyError -- if the given phoneme is not in the dictionary.
    """
    return phonemes[chars]
