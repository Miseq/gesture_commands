from threading import Timer

class SimpleTimer(Timer):
    def __init__(self):
        super(SimpleTimer, self).__init__(Timer)

    def run(self) -> None:
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)