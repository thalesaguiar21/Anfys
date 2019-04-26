import unittest
import tests.tests_mem_funcs as tests_mem_funcs
import tests.tests_regression as tests_regression
import tests.tests_tnorm as tests_tnorm
import tests.tests_tconorm as tests_tconorm

test_units = [tests_mem_funcs,
              tests_regression,
              tests_tnorm,
              tests_tconorm]

for utest in test_units:
    suite = unittest.TestLoader().loadTestsFromModule(utest)
    unittest.TextTestRunner(verbosity=2).run(suite)
