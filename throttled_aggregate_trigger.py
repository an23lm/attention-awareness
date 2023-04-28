from threading import Timer


class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


class ThrottledAggregateTrigger:
    def __init__(self, time, callback) -> None:
        self.queue = [False for i in range(0, 50)]
        self.time = time
        self.callback = callback
        self.timer = RepeatedTimer(1, self.timer_callback)

    def add(self, value):
        self.queue.pop(0)
        self.queue.append(value)

    def timer_callback(self):
        self.callback(self.queue.count(True) >= 25)

    def kill(self):
        self.timer.stop()
