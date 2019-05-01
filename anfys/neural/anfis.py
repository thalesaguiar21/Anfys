import numpy as np
import itertools
import pdb


class Tsukamoto:

    def __init__(self, fuzzy_set_size):
        pass


def create_rules(fuzzy_set_size, qtd_inputs):
    nodes_id = np.arange(fuzzy_set_size)
    return [x for x in itertools.product(nodes_id, repeat=qtd_inputs)]
