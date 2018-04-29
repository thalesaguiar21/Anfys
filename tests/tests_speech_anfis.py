from context import anfis, fuzz
import unittest2 as unittest
import numpy as np
import pdb


class TestBaseModel(unittest.TestCase):

    def setUp(self):
        self.outputs = [[1, 2, 3], [2, 2], [0.5, 3]]
        self.prec = [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
        self.anfis = anfis.BaseModel([3, 2, 2], self.prec, fuzz.BellTwo())

    def test_rules(self):
        self.setUp()
        expected = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                            [2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1]])
        num_rules = len(expected)
        rule_size = len(expected[0])
        for i in xrange(num_rules):
            for j in xrange(rule_size):
                self.assertEqual(
                    expected[i, j], self.anfis._rule_set[i, j]
                )

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
        arr = np.array([[0], [1]])
        for i in xrange(2):
            self.assertEqual(arr[i, 0], self.anfis._rule_set[i, 0])

    def test_rules_empty(self):
        self.anfis = anfis.BaseModel([], [], fuzz.BellTwo())
        self.assertSequenceEqual([()], self.anfis._rule_set)

    def test_min_expected(self):
        self.setUp()
        expected = [0.5, 1, 0.5, 1, 0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2]
        rs = self.anfis._min_operation(self.outputs)
        for i in xrange(rs.size):
            if expected[i] != rs[i]:
                self.fail('Error! Difference at element ' + str(i))

    def test_min_no_output(self):
        self.setUp()
        try:
            self.anfis._min_operation(None)
            self.fail('Computed min operation with None as outputs!')
        except ValueError:
            pass

    def test_prod_expected(self):
        self.setUp()
        expected = [1.0, 6.0, 1.0, 6.0, 2.0, 12.0,
                    2.0, 12.0, 3.0, 18.0, 3.0, 18.0]
        rs = self.anfis._product_operation(self.outputs)
        for i in xrange(rs.size):
            if expected[i] != rs[i]:
                self.fail('Error! Difference at element ' + str(i))

    def test_prod_no_output(self):
        self.setUp()
        try:
            self.anfis._product_operation(None)
            self.fail('Computed prod operation with None as outputs!')
        except ValueError:
            pass

    def test_sets(self):
        self.setUp()
        self.assertSequenceEqual(self.anfis._sets, [3, 5, 7])

    def test_different_size_param(self):
        self.setUp()
        prec = [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
        sets = [1, 2, 1]
        try:
            self.anfis = anfis.BaseModel(sets, prec, fuzz.BellTwo())
            self.fail('Created a model with different number of params and \
                       functions')
        except ValueError:
            pass


class TestTsukamoto(unittest.TestCase):

    def setUp(self):
        sets_size = [3, 2, 2]
        prec_params = [[3, 2], [3, 2], [3, 2], [3, 2],
                       [3, 2], [3, 2], [3, 2]]
        mem_func = fuzz.BellTwo()
        self.tsukamoto = anfis.TsukamotoModel(
            sets_size, prec_params, mem_func, fuzz.PiecewiseLogit()
        )

    def test_layer_1(self):
        self.setUp()
        entry = [4, 5, 6]
        expected = [[0.6411803884299546, 0.6411803884299546,
                    0.6411803884299546],
                    [0.36787944117144233, 0.36787944117144233],
                    [0.1690133154060661, 0.1690133154060661]]

        l1, l2, l3, l5 = self.tsukamoto.forward_pass(entry, 400)
        for line_exp, line_rs in zip(expected, l1):
            for elm, rs in zip(line_exp, line_rs):
                self.assertAlmostEqual(elm, rs)
                self.assertEqual(True, rs > 0 and rs < 1.0)

        for l2_out in l2:
            self.assertAlmostEqual(l2_out, 0.0398663678237249)

    def test_layer_output_range(self):
        self.setUp()
        entry = [4, 5, 6]
        l1, l2, l3, l5 = self.tsukamoto.forward_pass(entry, 400)
        for mem_degree in l1:
            for degree in mem_degree:
                self.assertEqual(True, degree > 0 and degree < 1.0)
        for elm in l2:
            self.assertEqual(True, elm > 0 and elm < 1.0)
        for elm in l3:
            self.assertEqual(True, elm > 0 and elm < 1.0)

    def test_find_consequents(self):
        self.setUp()
        sets_size = [1, 3]
        prec_params = [[3, 2], [3, 2], [3, 2],
                       [3, 2]]
        mem_func = fuzz.BellTwo()
        self.tsukamoto = anfis.TsukamotoModel(
            sets_size, prec_params, mem_func, fuzz.PiecewiseLogit()
        )
        threshold = 3e-1
        values = [1, 2, 3]
        weights = [3, 2, 1]
        expected = 10
        results = self.tsukamoto._find_consequents(values, weights, expected)
        values = [3, 2, 1]
        weights = [2, 2, 2]
        expected = 15
        results = self.tsukamoto._find_consequents(
            values, weights, expected, True
        )
        approximated_result = []
        for equation in self.tsukamoto.coef_matrix:
            total = 0
            for rs, coef in zip(results, equation):
                total += rs * coef
            approximated_result.append(total)

        for rs, expec in zip(approximated_result, self.tsukamoto.expected):
            if rs < expec - threshold:
                self.fail('Expected result on lse is far from expected!')
            if rs > expec + threshold:
                self.fail('Expected result on lse is far from expected!')

    def test_system_size(self):
        self.setUp()
        data = [([2, 3, 4], 10),
                ([2, 1, 4], 20),
                ([2, 3, 1], 25)]
        self.tsukamoto.learn_hybrid_online(data, max_epochs=1)
        sys_len = len(self.tsukamoto.coef_matrix)
        self.assertEqual(3, sys_len)

    def test_error_size(self):
        self.setUp()
        data = [([2, 3, 4], 10),
                ([2, 1, 4], 20),
                ([2, 3, 1], 25)]
        self.tsukamoto.learn_hybrid_online(data, max_epochs=1)
        erros_len = len(self.tsukamoto._errors)
        self.assertEqual(3, erros_len)
