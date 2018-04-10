from __future__ import print_function
import tests_mem_funcs
import tests_speech_utils
import tests_speech_anfis

# Testing membership functions
test_bell_three = tests_mem_funcs.TestBellThree()
test_bell_two = tests_mem_funcs.TestBellTwo()

print('Testing membership functions...')
test_bell_three.run_all()
test_bell_two.run_all()
print('OK!\n')

# Testing utils
print('Testing speech utils...', end='')
test_utils = tests_speech_utils.TestUtils()
test_utils.run_all()
print('OK!\n')

# Testing anfis
print('Testing anfis...', end='')
test_anfis = tests_speech_anfis.TestBaseModel()
test_anfis.run_all()
test_tsukamoto = tests_speech_anfis.TestTsukamoto()
test_tsukamoto.run_all()
print('OK!\n')
