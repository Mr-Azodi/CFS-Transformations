
import heapq
import itertools
import threading
import time
from dataclasses import dataclass

BASE_WEIGHT = 1024
TIME_SLICE = 10
NUM_WORKERS = 2


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
        self._heap = []                              # داده‌ی خصوصی (به‌جای درخت سرخ‌سیاه)
        self._seq = itertools.count()
        self._min_vruntime = 0.0
        self._stopped = False
        self._cond = threading.Condition()           # قفل مانیتور + متغیر شرط not_empty

    def enqueue(self, task):
        with self._cond:                             # enter monitor / exit خودکار
            task.vruntime = max(task.vruntime, self._min_vruntime)
            heapq.heappush(self._heap, (task.vruntime, next(self._seq), task))
            self._cond.notify()                      # signal(not_empty)

    def pick_next(self):
        with self._cond:
            while not self._heap and not self._stopped:   # همیشه while، نه if
                self._cond.wait()                    # wait(not_empty): قفل آزاد می‌شود
            if not self._heap:
                return None                          # خاموشی
            vr, _, task = heapq.heappop(self._heap)
            self._min_vruntime = max(self._min_vruntime, vr)
            return task

    def account(self, task, run):
        with self._cond:
            task.cpu_used += run
            task.slices += 1
            task.vruntime += run * (BASE_WEIGHT / task.weight)

    def shutdown(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()                  # broadcast(not_empty)


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


def worker(mon, wid, stats):
    while True:
        task = mon.pick_next()
        if task is None:
            return
        run = min(TIME_SLICE, task.remaining)
        time.sleep(run * 0.0005)                     # اجرا خارج از مانیتور (جلوگیری از بن‌بست)
        task.remaining -= run
        mon.account(task, run)
        done = task.remaining == 0
        all_done = stats.record(
            f"worker{wid}: {task.name:<4} ran {run:>2}  vruntime={task.vruntime:8.2f}", done)
        if not done:
            mon.enqueue(task)
        elif all_done:
            mon.shutdown()


def main():
    tasks = [Task("A", 1024, 100), Task("B", 2048, 100), Task("C", 512, 100), Task("D", 1024, 60)]
    mon, stats = CFSMonitor(), Stats(len(tasks))
    threads = [threading.Thread(target=worker, args=(mon, i, stats)) for i in range(NUM_WORKERS)]
    for t in threads:
        t.start()
    for t in tasks:
        mon.enqueue(t)
    for t in threads:
        t.join()

    print("\n".join(stats.log[:20]), "\n...")
    print("\nTask  weight  slices  cpu_used  final_vruntime")
    for t in tasks:
        print(f"{t.name:<5} {t.weight:>6} {t.slices:>7} {t.cpu_used:>9} {t.vruntime:>15.2f}")


if __name__ == "__main__":
    main()
