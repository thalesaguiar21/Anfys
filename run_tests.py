import unittest
import tests.tests_mem_funcs as tests_mem_funcs
import tests.tests_speech_utils as tests_speech_utils
import tests.tests_speech_anfis as tests_speech_anfis


suite_funcs = unittest.TestLoader().loadTestsFromModule(tests_mem_funcs)
unittest.TextTestRunner(verbosity=2).run(suite_funcs)

suite_utils = unittest.TestLoader().loadTestsFromModule(tests_speech_utils)
unittest.TextTestRunner(verbosity=2).run(suite_utils)

suite_anfis = unittest.TestLoader().loadTestsFromModule(tests_speech_anfis)
unittest.TextTestRunner(verbosity=2).run(suite_anfis)
