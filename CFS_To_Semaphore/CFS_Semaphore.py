
import heapq
import itertools
import threading
import time
from dataclasses import dataclass, field

BASE_WEIGHT = 1024      # وزن پایه (معادل nice = 0)
TIME_SLICE = 10         # طول هر برش زمانی (واحد فرضی)
NUM_WORKERS = 2         # تعداد «هسته»‌های شبیه‌سازی‌شده


@dataclass
class Task:
    name: str
    weight: int
    remaining: int                       # کار باقی‌مانده
    vruntime: float = 0.0
    cpu_used: int = 0
    slices: int = 0


class CFSSemaphore:

    def __init__(self):
        self._heap = []                              # (vruntime, شماره‌ی ترتیب, task)
        self._seq = itertools.count()                # شکستن تساوی vruntime
        self.min_vruntime = 0.0
        self.stopped = False
        self.tree_mutex = threading.Semaphore(1)     # دودویی: دسترسی انحصاری به درخت
        self.ready_count = threading.Semaphore(0)    # شمارشی: تعداد فرایندهای آماده
        self.stat_mutex = threading.Semaphore(1)     # دودویی: حفاظت از آمار مشترک

    def enqueue(self, task):
        self.tree_mutex.acquire()                    # wait(tree_mutex)
        task.vruntime = max(task.vruntime, self.min_vruntime)
        heapq.heappush(self._heap, (task.vruntime, next(self._seq), task))
        self.tree_mutex.release()                    # signal(tree_mutex)
        self.ready_count.release()                   # signal(ready_count)

    def pick_next(self):
        self.ready_count.acquire()                   # wait(ready_count): صف خالی = انتظار
        self.tree_mutex.acquire()                    # wait(tree_mutex)
        try:
            if not self._heap:                       # بیدارسازی برای خاموشی
                return None
            vr, _, task = heapq.heappop(self._heap)  # چپ‌ترین گره = کمترین vruntime
            self.min_vruntime = max(self.min_vruntime, vr)
            return task
        finally:
            self.tree_mutex.release()                # signal(tree_mutex)

    def shutdown(self, workers):
        self.stopped = True
        for _ in range(workers):
            self.ready_count.release()               # بیدار کردن همه‌ی نخ‌های منتظر


def worker(sched, wid, total, state, log):
    while True:
        task = sched.pick_next()
        if task is None:
            return
        run = min(TIME_SLICE, task.remaining)
        time.sleep(run * 0.0005)                     # شبیه‌سازی اجرای فرایند (خارج از قفل‌ها)
        task.remaining -= run
        task.cpu_used += run
        task.slices += 1
        task.vruntime += run * (BASE_WEIGHT / task.weight)   # فرایند سنگین‌تر کندتر پیر می‌شود

        sched.stat_mutex.acquire()
        log.append(f"worker{wid}: {task.name:<4} ran {run:>2}  vruntime={task.vruntime:8.2f}")
        if task.remaining == 0:
            state["finished"] += 1
            all_done = state["finished"] == total
        else:
            all_done = False
        sched.stat_mutex.release()

        if task.remaining > 0:
            sched.enqueue(task)
        elif all_done:
            sched.shutdown(NUM_WORKERS)


def main():
    tasks = [Task("A", 1024, 100), Task("B", 2048, 100), Task("C", 512, 100), Task("D", 1024, 60)]
    sched, state, log = CFSSemaphore(), {"finished": 0}, []
    threads = [threading.Thread(target=worker, args=(sched, i, len(tasks), state, log))
               for i in range(NUM_WORKERS)]
    for t in threads:
        t.start()
    for t in tasks:
        sched.enqueue(t)
    for t in threads:
        t.join()

    print("\n".join(log[:20]), "\n...")
    print("\nTask  weight  slices  cpu_used  final_vruntime")
    for t in tasks:
        print(f"{t.name:<5} {t.weight:>6} {t.slices:>7} {t.cpu_used:>9} {t.vruntime:>15.2f}")


if __name__ == "__main__":
    main()
