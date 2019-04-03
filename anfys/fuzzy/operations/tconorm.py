def fmax(a, b):
    return a if a > b else b


def probabilistic_sum(a, b):
    return a + b - a * b


def bounded_sum(a, b):
    result = a + b
    return result if result < 1.0 else 1.0


def drastic(a, b):
    result = None
    if a == 0:
        result = b
    elif b == 0:
        result = a
    else:
        result = 1.0
    return result


def nilpotent_max(a, b):
    result = fmax(a, b)
    return result if a + b < 1.0 else 1.0


def einstein_sum(a, b):
    return (a + b) / (1.0 + a * b)
