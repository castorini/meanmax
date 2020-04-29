import math
import random

import numpy as np


def split(lst, size):
    for idx in range(0, len(lst), size):
        yield lst[idx:idx + size]


def striped(indices, *lists):
    return [l[idx] for l, idx in zip(lists, indices)]


def cross(*lists, randomize=False):
    def increment():
        for idx, index in reversed(list(enumerate(indices))):
            if index < len(lists[idx]) - 1:
                indices[idx] += 1
                return True
            else:
                indices[idx] = 0
        return False

    if randomize:
        random_map = [list(range(len(x))) for x in lists]
        list(map(random.shuffle, random_map))

    if len(lists) == 0:
        raise StopIteration
    indices = [0] * len(lists)
    while True:
        yield striped([r[x] for r, x in zip(random_map, indices)] if randomize else indices, *lists)
        if not increment(): break


def flatten(lst, recursive=True, join=[]):
    flat_list = []
    for idx, x in enumerate(lst):
        if isinstance(x, list):
            flat_list.extend(flatten(x) if recursive else x)
            if idx != len(lst) - 1: flat_list.extend(join)
        else:
            flat_list.append(x)
    return flat_list


def chunk(lst, n):
    return split(lst, math.ceil(len(lst) / n))


def interlacing_repeat(lst, n):
    for x in lst:
        for _ in range(n):
            yield x


def interlacing_expand(lst, n, callback):
    for idx, x in enumerate(lst):
        for _ in range(n):
            yield x
            x = callback(idx, x)
