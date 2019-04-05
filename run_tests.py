import unittest
import tests.tests_mem_funcs as tests_mem_funcs
import tests.tests_regression as tests_regression
import tests.tests_tnorm as tests_tnorm
import tests.tests_tconorm as tests_tconorm


suite_funcs = unittest.TestLoader().loadTestsFromModule(tests_mem_funcs)
unittest.TextTestRunner(verbosity=2).run(suite_funcs)

suite_reg = unittest.TestLoader().loadTestsFromModule(tests_regression)
unittest.TextTestRunner(verbosity=2).run(suite_reg)

suite_tnorm = unittest.TestLoader().loadTestsFromModule(tests_tnorm)
unittest.TextTestRunner(verbosity=2).run(suite_tnorm)

suite_tconorm = unittest.TestLoader().loadTestsFromModule(tests_tconorm)
unittest.TextTestRunner(verbosity=2).run(suite_tconorm)

# suite_anfis = unittest.TestLoader().loadTestsFromModule(tests_speech_anfis)
# unittest.TextTestRunner(verbosity=2).run(suite_anfis)
