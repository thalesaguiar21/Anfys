from context import anfis
import unittest2 as unittest


class TestRuleCreation(unittest.TestCase):

    def __init__(self):
        self.outputs = [[1, 2, 3], [2, 2], [0.5, 3]]
        self.rules_size = [3, 2, 2]

    def test_rules(self):
        expected = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0),
                    (1, 0, 1), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 0, 1),
                    (2, 1, 0), (2, 1, 1)]
        res = anfis._create_rules(self.rules_size)
        self.assertSequenceEqual(res, expected)

    def test_rules_none(self):
        try:
            anfis._create_rules(None)
            self.fail('Rules created with no sets!')
        except ValueError:
            pass

    def test_rules_one(self):
        self.rules_size = [2]
        res = anfis._create_rules(self.rules_size)
        self.assertSequenceEqual([(0,), (1,)], res)

    def test_rules_empty(self):
        self.assertSequenceEqual([()], anfis._create_rules([]))

    def run_all(self):
        self.test_rules()
        self.test_rules_none()
        self.test_rules_one()
        self.test_rules_empty()


test_anfis = TestRuleCreation()
test_anfis.run_all()
