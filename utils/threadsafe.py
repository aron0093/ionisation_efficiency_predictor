import threading

class threadsafe_iter:

    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):

    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):

        return threadsafe_iter(f(*a, **kw))

    return g

if __name__ == "__main__":

    @threadsafe_generator
    def gen():
        print('generator initiated')
        idx = 0
        while True:
            yield x_train[:32], y_train[:32]
            print('generator yielded a batch %d' % idx)
            idx += 1

    tr_gen = gen()