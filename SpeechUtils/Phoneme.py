from random import random


phonemeList = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER',
    'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z',
    'ZH'
]

thresholds = [random()] * len(phonemeList)
for i in range(1, len(thresholds)):
    thresholds[i] = thresholds[i - 1] + random()

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


def step(x):
    """ Computes if the given value correponds to any phoneme

    Parameters
    ----------
    x : double
        A value to be tested

    Returns
    -------
    phn : str
        An string with the corresponding phoneme, or '' if the value does not
        correspond to any phoneme.
    """
    phn = ''
    for i in range(len(thresholds)):
        if x <= thresholds[i]:
            phn = phonemeList[i]
            break
    return phn


def phone_threshold(phone):
    """ Gets the threshold value for an phoneme

    Parameters
    ----------
    phone : str
        A string with an representation of the phoneme

    Returns
    -------
    threshold : double
        The maximum value where the given phoneme would be inferred
    """
    return thresholds[index(phone)]
