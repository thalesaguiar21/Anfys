import numpy as np
from itertools import product
from anfys.neural.anfis import Sugeno
import anfys.lse as regression


def hybrid_online(anfis, entry, output):
    first_three_layers = _half_forward_pass(entry, output)
    _update_consequent_parameters(anfis, first_three_layers, entry, output)
    l4 = _update_consequent_parameters(anfis)
    l5 = prediction(l4)
    return l5


def _half_forward_pass(anfis, entry, output):
    # Forward inputs until the third layer
    layer1 = _fuzzysets_membership_degrees(anfis, entry)
    layer2 = _rules_fire_strength(layer1)
    layer3 = _averaged_fire_strength(layer2)
    return layer1, layer2, layer3


def _fuzzysets_membership_degrees(anfis, inputs):
    l1size = anfis.l1size()
    layer1 = np.zeros((l1size, anfis.qtd_mfs))
    param_range = np.arange(0, l1size + 1, anfis.qtd_mfs)
    for feat, subset, out in zip(inputs, anfis.sets, range(l1size)):
        at, untill = param_range[out], param_range[out + 1]
        output = subset.evaluate(feat, anfis.prem_params[at:untill])
        layer1[out] = output
    return layer1


def _rules_fire_strength(anfis, mdegrees):
    nodes_id = np.arange(anfis.qtd_mfs)
    # Create every combination for the given
    layer2 = []
    for mf in product(nodes_id, repeat=anfis.qtd_inputs):
        rule = [mdegrees[n_set, mf[n_set]] for n_set in range(anfis.qtd_sets)]
        layer2.append(np.prod(rule))
    return np.array(layer2)


def _averaged_fire_strength(fire_strengths):
    total_strength = np.sum(fire_strengths)
    return [rstrength / total_strength for rstrength in fire_strengths]


def _update_consequent_parameters(anfis, entry, output, weights):
    if isinstance(anfis, Sugeno):
        _update_consequent_parameters(anfis, entry, output, weights)
    else:
        raise ValueError(
            'Unkown {} model passed to learn.update_parameters!'.format(anfis))


def _solve_consequent_system(anfis, entry, output, weights):
    column_weights = np.array([weights]).T
    coefs = column_weights.dot(entry)
    anfis.add_linsys_equation(coefs, output)
    anfis.cons_params = regression.solve(
        anfis.linsys_coefs, anfis.linsys_resul)
