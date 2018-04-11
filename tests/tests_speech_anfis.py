from context import anfis, fuzz
import unittest2 as unittest
import pprint


class TestBaseModel(unittest.TestCase):

    def __init__(self):
        self.outputs = [[1, 2, 3], [2, 2], [0.5, 3]]
        self.prec = [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
        self.anfis = anfis.BaseModel([3, 2, 2], self.prec, fuzz.BellTwo())

    def setup(self):
        self.outputs = [[1, 2, 3], [2, 2], [0.5, 3]]
        self.prec = [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
        self.anfis = anfis.BaseModel([3, 2, 2], self.prec, fuzz.BellTwo())

    def test_rules(self):
        self.setup()
        expected = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0),
                    (1, 0, 1), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 0, 1),
                    (2, 1, 0), (2, 1, 1)]
        self.assertSequenceEqual(expected, self.anfis._rule_set)

    def test_rules_none(self):
        self.prec = [[1, 2], [1, 2], [1, 2], [1, 2]]
        try:
            self.anfis = anfis.BaseModel(None, self.prec, fuzz.BellTwo())
            self.fail('Rules created with no sets!')
        except ValueError:
            pass

        try:
            self.anfis = anfis.BaseModel([-1, 2, 3], self.prec, fuzz.BellTwo())
            self.fail('Rules created with negative size!')
        except ValueError:
            pass

        try:
            self.anfis = anfis.BaseModel(
                [-1, 2, None], self.prec, fuzz.BellTwo())
            self.fail('Rules created with no sets!')
        except ValueError:
            pass

    def test_rules_one(self):
        self.anfis = anfis.BaseModel([2], [[1, 2], [1, 2]], fuzz.BellTwo())
        self.assertSequenceEqual([(0,), (1,)], self.anfis._rule_set)

    def test_rules_empty(self):
        self.anfis = anfis.BaseModel([], [], fuzz.BellTwo())
        self.assertSequenceEqual([()], self.anfis._rule_set)

    def test_min_expected(self):
        self.setup()
        expected = [0.5, 1, 0.5, 1, 0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2]
        self.assertSequenceEqual(
            expected, self.anfis._min_operation(self.outputs))

    def test_min_no_output(self):
        self.setup()
        try:
            self.anfis._min_operation(None)
            self.fail('Computed min operation with None as outputs!')
        except ValueError:
            pass

    def test_prod_expected(self):
        self.setup()
        expected = [1.0, 6.0, 1.0, 6.0, 2.0, 12.0,
                    2.0, 12.0, 3.0, 18.0, 3.0, 18.0]
        self.assertSequenceEqual(
            expected, self.anfis._product_operation(self.outputs))

    def test_prod_no_output(self):
        self.setup()
        try:
            self.anfis._product_operation(None)
            self.fail('Computed prod operation with None as outputs!')
        except ValueError:
            pass

    def test_sets(self):
        self.setup()
        self.assertSequenceEqual(self.anfis._sets, [3, 5, 7])

    def test_different_size_param(self):
        sets = [1, 2, 1]
        try:
            self.anfis = anfis.BaseModel(sets, self.prec. fuzz.BellTwo())
            self.fail('Created a model with different number of params and \
                       functions')
        except ValueError:
            pass

    def run_all(self):
        self.test_rules()
        self.test_rules_none()
        self.test_rules_one()
        self.test_rules_empty()
        self.test_min_expected()
        self.test_min_no_output()
        self.test_prod_expected()
        self.test_prod_no_output()
        self.test_sets()


class TestTsukamoto(unittest.TestCase):

    def __init__(self):
        sets_size = [3, 2, 2]
        prec_params = [[3, 2], [3, 2], [3, 2], [3, 2],
                       [3, 2], [3, 2], [3, 2]]
        mem_func = fuzz.BellTwo()
        self.tsukamoto = anfis.TsukamotoModel(sets_size, prec_params, mem_func)

    def setup(self):
        sets_size = [3, 2, 2]
        prec_params = [[3, 2], [3, 2], [3, 2], [3, 2],
                       [3, 2], [3, 2], [3, 2]]
        mem_func = fuzz.BellTwo()
        self.tsukamoto = anfis.TsukamotoModel(sets_size, prec_params, mem_func)

    def test_layer_1(self):
        self.setup()
        pp = pprint.PrettyPrinter()
        entry = [4, 5, 6]
        expected = [[0.6411803884299546, 0.6411803884299546,
                    0.6411803884299546],
                    [0.36787944117144233, 0.36787944117144233],
                    [0.1690133154060661, 0.1690133154060661]]

        layer_1, layer_2 = self.tsukamoto.forward_pass(entry, 400)
        for line_exp, line_rs in zip(expected, layer_1):
            for elm, rs in zip(line_exp, line_rs):
                self.assertAlmostEqual(elm, rs)
        for l2_out in layer_2:
            self.assertAlmostEqual(l2_out, 0.0398663678237249)

    def run_all(self):
        self.test_layer_1()
