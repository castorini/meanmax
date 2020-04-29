class WrappedObject(object):

    def __init__(self, x):
        self.x = x

    def __hash__(self):
        return id(self.x)

    def __eq__(self, other):
        return id(other) == id(self)


def id_wrap(x):
    return WrappedObject(x)