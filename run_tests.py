import unittest
import tests.tests_mem_funcs as tests_mem_funcs
import tests.tests_lse as tests_lse
import tests.tests_tnorm as tests_tnorm
import tests.tests_tconorm as tests_tconorm
import tests.tests_neural as tests_neural

test_units = [tests_mem_funcs,
              tests_lse,
              tests_tnorm,
              tests_tconorm,
              tests_neural]

for utest in test_units:
    suite = unittest.TestLoader().loadTestsFromModule(utest)
    unittest.TextTestRunner(verbosity=2).run(suite)
