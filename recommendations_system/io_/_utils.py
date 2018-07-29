def flatten(lst):
    return [item for sublist in lst for item in sublist]


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False
