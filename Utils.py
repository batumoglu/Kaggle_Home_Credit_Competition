

class Profiler(object):
    def __init__(self):
        self._start_ = None
        self._end_ = None

    def Start(self):
        from time import time
        self._start_ = time()

    def End(self):
        from time import time
        self._end_ = time()

    @property
    def ElapsedSeconds(self):
        if self._start_ is not None and self._end_ is not None:
            return (int)(self._end_ - self._start_)
        return -1

    @property
    def ElapsedMinutes(self):
        if self._start_ is not None and self._end_ is not None:
            return (int)((self._end_ - self._start_)/60)
        return -1




    