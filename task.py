import threading
import time
from queue import Queue

class Runner:
    def __init__(self, max_threads=4):
        self.max_threads = max_threads
        self.task_queue = Queue()
        self.semaphore = threading.Semaphore(max_threads)


    def process_task(self, task):
        task()
        self.semaphore.release()

    def add_task(self, task):
        self.task_queue.put(task)

        if self.semaphore.acquire(blocking=False):
            task = self.task_queue.get()
            t = threading.Thread(target=self.process_task, args=(task,))
            t.start()
        else:
            print("Waiting for an available thread...")

    def start(self):
        self.task_queue.join()

if __name__ == '__main__':
    runner = Runner(2)

    def sleep():
        time.sleep(3)
        print('done 1')

    for i in range(5):
        runner.add_task(sleep)

    runner.start()
