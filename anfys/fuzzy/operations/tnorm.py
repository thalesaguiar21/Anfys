def fmin(a, b):
    return a if a < b else b


def prod(a, b):
    return a * b


def lukasiewicz(a, b):
    return max(a + b - 1, 0.0)


def drastic(a, b):
    result = None
    if a == 1:
        result = b
    elif b == 1:
        result = a
    else:
        result = 0
    return result


def nilpotent(a, b):
    result = None
    if a + b > 1:
        result = fmin(a, b)
    else:
        result = 0
    return result


def hamacher(a, b):
    result = None
    if a == b and b == 0:
        result = 0
    else:
        prod = a * b
        result = prod / (a + b - prod)
    return result
