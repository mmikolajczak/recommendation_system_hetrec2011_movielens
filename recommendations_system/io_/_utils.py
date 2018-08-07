import pickle


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def load_ffm(path):
    with open(path, 'r', newline='\n') as f:
        data = f.readlines()
    return data


def save_ffm(data, path):
    with open(path, 'w', newline='\n') as f:
        f.writelines(data)


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
