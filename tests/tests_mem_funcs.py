from .context import memfuncs
import unittest


class TestBellTwo(unittest.TestCase):

    def setUp(self):
        self.bellTwo = memfuncs.BellTwo()
        self.value = 4
        self.a = 3
        self.b = 2

    def test_mem_degree(self):
        self.setUp()
        res = self.bellTwo.membership_degree(self.value, self.a, self.b)
        self.assertAlmostEqual(res, 0.64118038843, 13)

    def test_mem_degree_three(self):
        self.setUp()
        res = self.bellTwo.membership_degree(self.value, self.a, self.b, 40)
        self.assertAlmostEqual(res, 0.64118038843, 13)

    def test_mem_degree_none(self):
        self.setUp()
        try:
            self.bellTwo.membership_degree(None, None, None)
            self.fail('Mem degree computed with a None argument')
        except TypeError:
            pass

    def test_mem_degree_zero(self):
        self.setUp()
        params = [0.022528536199578036, 1.5046653242073884]
        rs = self.bellTwo.membership_degree(0, *params)
        self.assertEqual(rs, 1e-10)

    def test_mem_degree_zerodivision(self):
        self.setUp()
        try:
            self.bellTwo.membership_degree(self.value, 0.0, self.b)
        except ZeroDivisionError:
            pass

    def test_derivative_on_a(self):
        self.setUp()
        res = self.bellTwo.partial(self.value, 'a', self.a, self.b)
        self.assertAlmostEqual(res, 0.18997937435, 12)
        res = self.bellTwo.partial(self.value, 'a', self.a, self.b, 40)
        self.assertAlmostEqual(res, 0.18997937435, 12)

    def test_derivative_on_b(self):
        self.setUp()
        res = self.bellTwo.partial(self.value, 'b', self.a, self.b)
        self.assertAlmostEqual(res, 0.284969061524, 12)
        res = self.bellTwo.partial(self.value, 'b', self.a, self.b, 40)
        self.assertAlmostEqual(res, 0.284969061524, 12)

    def test_derivative_on_other(self):
        self.setUp()
        res = self.bellTwo.partial(self.value, 'c', self.a, self.b)
        self.assertAlmostEqual(res, 0.0, 13)
        res = self.bellTwo.partial(self.value, 'c', self.a, self.b, 40)
        self.assertAlmostEqual(res, 0.0, 13)

    def test_derivative_none(self):
        self.setUp()
        try:
            self.bellTwo.partial(None, None, None, None)
            self.fail('Mem derivative computed with None')
        except TypeError:
            pass


class TestBellThree(unittest.TestCase):

    def setUp(self):
        self.bellThree = memfuncs.BellThree()
        self.value = 4
        self.a = 3
        self.b = 2
        self.c = 2

    def test_mem_degree(self):
        self.setUp()
        res = self.bellThree.membership_degree(
            self.value, self.a, self.b, self.c
        )
        self.assertAlmostEqual(res, 0.835051546392, 12)

    def test_mem_degree_none(self):
        self.setUp()
        try:
            self.bellThree.membership_degree(None, None, None, None)
            self.fail('Mem degree computed with a None argument')
        except ValueError:
            pass

    def test_mem_degree_zerodivision(self):
        self.setUp()
        try:
            self.bellThree.membership_degree(self.value, 0.0, self.b, self.c)
        except ValueError:
            pass

    def test_derivative_on_a(self):
        self.setUp()
        res = self.bellThree.partial(
            self.value, 'a', self.a, self.b, self.c
        )
        self.assertAlmostEqual(res, 0.183653948347, 12)

    def test_derivative_on_b(self):
        self.setUp()
        res = self.bellThree.partial(
            self.value, 'b', self.a, self.b, self.c
        )
        self.assertAlmostEqual(res, 0.1116979020317, 12)

    def test_derivative_on_c(self):
        self.setUp()
        res = self.bellThree.partial(
            self.value, 'c', self.a, self.b, self.c
        )
        self.assertAlmostEqual(res, 0.275480922521, 12)

    def test_derivative_none(self):
        self.setUp()
        try:
            self.bellThree.partial(None, None, None, None)
        except ValueError:
            pass


class TestPiecewiseLogit(unittest.TestCase):

    def setUp(self):
        self.plogit = memfuncs.PiecewiseLogit()
        self.params = [2, 10]
        self.value = 3

    def test_mem_degree(self):
        self.setUp()
        values = [0.000001, 0.3, 0.15, 0.99, 4, -1]
        expec = [2.0000078400001557, 4.399999887999996, 3.199999863999996,
                 9.9199999984, 10, 2]
        for value, e in zip(values, expec):
            self.assertAlmostEqual(
                e, self.plogit.membership_degree(value, *self.params), 6
            )

    def test_derivative(self):
        self.setUp()
        self.value = 0.3
        rs = []
        for variable in ['a', 'b', 'c']:
            rs.append(self.plogit.partial(
                self.value, variable, *self.params)
            )
