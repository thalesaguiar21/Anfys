from __future__ import print_function
import tests_mem_funcs
import tests_speech_utils
import tests_speech_anfis

# Testing membership functions
test_bell_three = tests_mem_funcs.TestBellThree()
test_bell_two = tests_mem_funcs.TestBellTwo()
test_plogit = tests_mem_funcs.TestPiecewiseLogit()

print('Testing membership functions...')
test_bell_three.run_all()
test_bell_two.run_all()
test_plogit.run_all()

# Testing utils
print('Testing speech utils...')
test_utils = tests_speech_utils.TestUtils()
test_utils.run_all()

# Testing anfis
print('Testing anfis...')
test_anfis = tests_speech_anfis.TestBaseModel()
test_anfis.run_all()
test_tsukamoto = tests_speech_anfis.TestTsukamoto()
test_tsukamoto.run_all()

print('\n\nDone!\n')
