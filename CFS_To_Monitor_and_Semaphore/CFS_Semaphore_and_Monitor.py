
import heapq
import itertools
import threading
import time
from dataclasses import dataclass

BASE_WEIGHT = 1024
TIME_SLICE = 10
NUM_WORKERS = 2
QUEUE_CAPACITY = 3      # حداکثر فرایندهای هم‌زمان در سیستم (عمداً کم تا اثر slots دیده شود)


@dataclass
class Task:
    name: str
    weight: int
    remaining: int
    vruntime: float = 0.0
    cpu_used: int = 0
    slices: int = 0


class CFSMonitor:

    def __init__(self):
        self._heap = []
        self._seq = itertools.count()
        self._min_vruntime = 0.0
        self._lock = threading.Lock()                # قفل مانیتور

    def insert(self, task):
        with self._lock:
            task.vruntime = max(task.vruntime, self._min_vruntime)
            heapq.heappush(self._heap, (task.vruntime, next(self._seq), task))

    def remove_leftmost(self):
        with self._lock:
            if not self._heap:
                return None
            vr, _, task = heapq.heappop(self._heap)
            self._min_vruntime = max(self._min_vruntime, vr)
            return task

    def account(self, task, run):
        with self._lock:
            task.cpu_used += run
            task.slices += 1
            task.vruntime += run * (BASE_WEIGHT / task.weight)


class Scheduler:
    def __init__(self, capacity):
        self.monitor = CFSMonitor()
        self.slots = threading.Semaphore(capacity)   # جایگاه خالی صف
        self.ready = threading.Semaphore(0)          # فرایند آماده

    def submit(self, task):
        self.slots.acquire()                         # wait(slots): سیستم پر = انتظار (بیرون مانیتور)
        self.requeue(task)

    def requeue(self, task):                         # بازگشت پس از یک برش زمانی (بدون slot جدید)
        self.monitor.insert(task)
        self.ready.release()                         # signal(ready)

    def complete(self):                              # پایان کار فرایند: آزاد کردن slot
        self.slots.release()                         # signal(slots)

    def pick_next(self):
        self.ready.acquire()                         # wait(ready): صف خالی = انتظار (بیرون مانیتور)
        return self.monitor.remove_leftmost()        # None = پیام خاموشی

    def shutdown(self, workers):
        for _ in range(workers):
            self.ready.release()


class Stats:
    def __init__(self, total):
        self._lock = threading.Lock()
        self.total, self.finished, self.log = total, 0, []

    def record(self, line, done):
        with self._lock:
            self.log.append(line)
            if done:
                self.finished += 1
            return self.finished == self.total


def worker(sched, wid, stats):
    while True:
        task = sched.pick_next()
        if task is None:
            return
        run = min(TIME_SLICE, task.remaining)
        time.sleep(run * 0.0005)                     # اجرا خارج از مانیتور و سمافورها
        task.remaining -= run
        sched.monitor.account(task, run)
        done = task.remaining == 0
        all_done = stats.record(
            f"worker{wid}: {task.name:<4} ran {run:>2}  vruntime={task.vruntime:8.2f}", done)
        if not done:
            sched.requeue(task)
        else:
            sched.complete()
            if all_done:
                sched.shutdown(NUM_WORKERS)


def main():
    tasks = [Task("A", 1024, 100), Task("B", 2048, 100), Task("C", 512, 100),
             Task("D", 1024, 60), Task("E", 2048, 80), Task("F", 1024, 40)]
    sched, stats = Scheduler(QUEUE_CAPACITY), Stats(len(tasks))
    threads = [threading.Thread(target=worker, args=(sched, i, stats)) for i in range(NUM_WORKERS)]
    for t in threads:
        t.start()
    for t in tasks:                                  # تولیدکننده: ورود به صف محدود
        sched.submit(t)
    for t in threads:
        t.join()

    print("\n".join(stats.log[:20]), "\n...")
    print("\nTask  weight  slices  cpu_used  final_vruntime")
    for t in tasks:
        print(f"{t.name:<5} {t.weight:>6} {t.slices:>7} {t.cpu_used:>9} {t.vruntime:>15.2f}")


if __name__ == "__main__":
    main()
