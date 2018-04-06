import tests_mem_funcs
import tests_speech_utils

# Testing membership functions
test_bell_three = tests_mem_funcs.TestBellThree()
test_bell_two = tests_mem_funcs.TestBellTwo()

test_bell_three.run_all()
test_bell_two.run_all()

# Testing utils
test_utils = tests_speech_utils.TestUtils()
test_utils.run_all()
