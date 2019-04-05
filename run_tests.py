import unittest
import tests.tests_mem_funcs as tests_mem_funcs
import tests.tests_regression as tests_regression
import tests.tests_speech_anfis as tests_speech_anfis
import tests.tests_utils as tests_utils
import tests.tests_tnorm as tests_tnorm


suite_funcs = unittest.TestLoader().loadTestsFromModule(tests_mem_funcs)
unittest.TextTestRunner(verbosity=2).run(suite_funcs)

suite_reg = unittest.TestLoader().loadTestsFromModule(tests_regression)
unittest.TextTestRunner(verbosity=2).run(suite_reg)

suite_utils = unittest.TestLoader().loadTestsFromModule(tests_utils)
unittest.TextTestRunner(verbosity=2).run(suite_utils)

suite_tnorm = unittest.TestLoader().loadTestsFromModule(tests_tnorm)
unittest.TextTestRunner(verbosity=2).run(suite_tnorm)

# suite_anfis = unittest.TestLoader().loadTestsFromModule(tests_speech_anfis)
# unittest.TextTestRunner(verbosity=2).run(suite_anfis)
