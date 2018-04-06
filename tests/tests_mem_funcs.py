from context import fuzz
import unittest2 as unittest


class TestBellTwo(unittest.TestCase):

    def __init__(self):
        self.bellTwo = fuzz.BellTwo()
        self.value = 0
        self.a = 0
        self.b = 0

    def __setup(self):
        self.bellTwo = fuzz.BellTwo()
        self.value = 4
        self.a = 3
        self.b = 2

    def __test_mem_degree(self):
        self.__setup()
        res = self.bellTwo.membership_degree(self.value, self.a, self.b)
        self.assertAlmostEqual(res, 0.64118038843, 13)

    def __test_mem_degree_three(self):
        self.__setup()
        res = self.bellTwo.membership_degree(self.value, self.a, self.b, 40)
        self.assertAlmostEqual(res, 0.64118038843, 13)

    def __test_mem_degree_none(self):
        self.__setup()
        try:
            self.bellTwo.membership_degree(None, None, None)
            self.fail('Mem degree computed with a None argument')
        except TypeError:
            pass

    def __test_mem_degree_zerodivision(self):
        self.__setup()
        try:
            self.bellTwo.membership_degree(self.value, 0.0, self.b)
        except ZeroDivisionError:
            pass

    def __test_derivative_on_a(self):
        self.__setup()
        res = self.bellTwo.derivative_at(self.value, 'a', self.a, self.b)
        self.assertAlmostEqual(res, 0.18997937435, 12)
        res = self.bellTwo.derivative_at(self.value, 'a', self.a, self.b, 40)
        self.assertAlmostEqual(res, 0.18997937435, 12)

    def __test_derivative_on_b(self):
        self.__setup()
        res = self.bellTwo.derivative_at(self.value, 'b', self.a, self.b)
        self.assertAlmostEqual(res, 0.284969061524, 12)
        res = self.bellTwo.derivative_at(self.value, 'b', self.a, self.b, 40)
        self.assertAlmostEqual(res, 0.284969061524, 12)

    def __test_derivative_on_other(self):
        self.__setup()
        res = self.bellTwo.derivative_at(self.value, 'c', self.a, self.b)
        self.assertAlmostEqual(res, 0.0, 13)
        res = self.bellTwo.derivative_at(self.value, 'c', self.a, self.b, 40)
        self.assertAlmostEqual(res, 0.0, 13)

    def __test_derivative_none(self):
        self.__setup()
        try:
            self.bellTwo.derivative_at(None, None, None, None)
            self.fail('Mem derivative computed with None')
        except TypeError:
            pass

    def run_all(self):
        self.__test_mem_degree()
        self.__test_mem_degree_three()
        self.__test_mem_degree_none()
        self.__test_mem_degree_zerodivision()
        self.__test_derivative_on_a()
        self.__test_derivative_on_b()
        self.__test_derivative_on_other()
        self.__test_derivative_none()


class TestBellThree(unittest.TestCase):

    def __init__(self):
        self.bellThree = fuzz.BellThree()
        self.value = 0
        self.a = 0
        self.b = 0
        self.c = 0

    def __setup(self):
        self.bellThree = fuzz.BellThree()
        self.value = 4
        self.a = 3
        self.b = 2
        self.c = 2

    def __test_mem_degree(self):
        self.__setup()
        res = self.bellThree.membership_degree(
            self.value, self.a, self.b, self.c
        )
        self.assertAlmostEqual(res, 0.835051546392, 12)

    def __test_mem_degree_none(self):
        self.__setup()
        try:
            self.bellThree.membership_degree(None, None, None, None)
            self.fail('Mem degree computed with a None argument')
        except ValueError:
            pass

    def __test_mem_degree_zerodivision(self):
        self.__setup()
        try:
            self.bellThree.membership_degree(self.value, 0.0, self.b, self.c)
        except ZeroDivisionError:
            pass

    def __test_derivative_on_a(self):
        self.__setup()
        res = self.bellThree.derivative_at(
            self.value, 'a', self.a, self.b, self.c
        )
        self.assertAlmostEqual(res, 0.183653948347, 12)

    def __test_derivative_on_b(self):
        self.__setup()
        res = self.bellThree.derivative_at(
            self.value, 'b', self.a, self.b, self.c
        )
        self.assertAlmostEqual(res, 0.111697902032, 12)

    def __test_derivative_on_other(self):
        self.__setup()
        res = self.bellThree.derivative_at(
            self.value, 'c', self.a, self.b, self.c
        )
        self.assertAlmostEqual(res, 0.275480922521, 12)

    def __test_derivative_none(self):
        self.__setup()
        try:
            self.bellThree.derivative_at(None, None, None, None)
        except ValueError:
            pass

    def run_all(self):
        self.__test_mem_degree()
        self.__test_mem_degree_none()
        self.__test_mem_degree_zerodivision()
        self.__test_derivative_on_a()
        self.__test_derivative_on_b()
        self.__test_derivative_on_other()
        self.__test_derivative_none()
