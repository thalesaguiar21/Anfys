from BellTwo import BellTwo
from BellThree import BellThree


b2 = BellTwo()
b3 = BellThree()

value = 4
a = 3
b = 2
c = 2


def is_closed(value, expected, threshold):
    return abs(value - expected) <= threshold


def test_bell_two():
    limit = 1e15
    total = 4.0
    hits = 4

    expected = 0.64118038843
    result = b2.membership_degree(value, a, b)
    in_range = is_closed(result, expected, limit)
    print 'Memebership degree... {}'.format(
        'Passed!' if in_range else 'Wrong!'
    )
    if not in_range:
        hits -= 1
        print 'Expected {}, result was {} '.format(expected, result)

    expected = 0.18997937435
    result = b2.derivative_at(value, 'a', a, b)
    in_range = is_closed(result, expected, limit)
    print 'Derivative at {} relative to \'a\'... {}'.format(
        value, 'Passed!' if in_range else 'Wrong!'
    )
    if not in_range:
        hits -= 1
        print 'Expected {}, result was {} '.format(expected, result)

    expected = 0.284969061524
    result = b2.derivative_at(value, 'b', a, b)
    in_range = is_closed(result, expected, limit)
    print 'Derivative at {} relative to \'b\'... {}'.format(
        value, 'Passed!' if in_range else 'Wrong!'
    )
    if not in_range:
        hits -= 1
        print 'Expected {}, result was {} '.format(expected, result)

    expected = 0.0
    result = b2.derivative_at(value, 'c', a, b)
    in_range = is_closed(result, expected, limit)
    print 'Derivative at {} relative to \'c\'... {}'.format(
        value, 'Passed!' if in_range else 'Wrong!'
    )
    if not in_range:
        hits -= 1
        print 'Expected {}, result was {} '.format(expected, result)

    print 'Tests finished, total is {}%'.format(hits / total * 100)


def test_bell_three():
    limit = 1e15
    total = 4.0
    hits = 4

    expected = 0.835051546392
    result = b3.membership_degree(value, a, b, c)
    in_range = is_closed(result, expected, limit)
    print 'Memebership degree... {}'.format(
        'Passed!' if in_range else 'Wrong!'
    )
    if not in_range:
        hits -= 1
        print 'Expected {}, result was {} '.format(expected, result)

    expected = 0.183653948347
    result = b3.derivative_at(value, 'a', a, b, c)
    in_range = is_closed(result, expected, limit)
    print 'Derivative at {} relative to \'a\'... {}'.format(
        value, 'Passed!' if in_range else 'Wrong!'
    )
    if not in_range:
        hits -= 1
        print 'Expected {}, result was {} '.format(expected, result)

    expected = 0.111697902032
    result = b3.derivative_at(value, 'b', a, b, c)
    in_range = is_closed(result, expected, limit)
    print 'Derivative at {} relative to \'b\'... {}'.format(
        value, 'Passed!' if in_range else 'Wrong!'
    )
    if not in_range:
        hits -= 1
        print 'Expected {}, result was {} '.format(expected, result)

    expected = 0.275480922521
    result = b3.derivative_at(value, 'c', a, b, c)
    in_range = is_closed(result, expected, limit)
    print 'Derivative at {} relative to \'c\'... {}'.format(
        value, 'Passed!' if in_range else 'Wrong!'
    )
    if not in_range:
        hits -= 1
        print 'Expected {}, result was {} '.format(expected, result)

    print 'Tests finished, total is {}%'.format(hits / total * 100)


print 'Tests on Bell Two: '
test_bell_two()
print 100 * '-'
print 'Tests on Bell Three'
test_bell_three()
