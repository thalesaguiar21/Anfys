from .context import anfys
import anfys.neural.anfis as anfis
import unittest


class TestRules(unittest.TestCase):

    def for_input_and_mfs(self, qtd_inps, qtd_mfs):
        self.qtd_inp = qtd_inps
        self.qtd_mfs = qtd_mfs

    def generate(self):
        self.rules = anfis.create_rules(self.qtd_mfs, self.qtd_inp)

    def number_of_rules_is(self, size):
        self.assertEqual(size, len(self.rules))

    def rules_are(self, rules):
        self.assertSequenceEqual(self.rules, rules)

    def test_size_3inputs_3mfuncs(self):
        self.for_input_and_mfs(3, 3)
        self.generate()
        self.number_of_rules_is(27)

    def test_size_0inputs(self):
        self.for_input_and_mfs(0, 1)
        self.generate()
        self.number_of_rules_is(1)  # Empty set of rules

    def test_0inputs(self):
        self.for_input_and_mfs(0, 1)
        self.generate()
        self.rules_are([()])

    def test_size_0mfuncs(self):
        self.for_input_and_mfs(1, 0)
        self.generate()
        self.number_of_rules_is(0)

    def test_0mfuncs(self):
        self.for_input_and_mfs(1, 0)
        self.generate()
        self.rules_are([])
