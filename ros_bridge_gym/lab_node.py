from uuid import uuid4
from queue import Queue


class Task(object):
    def __init__(self, x, y, uuid=None) -> None:
        super().__init__()
        self.uuid = str(uuid4()) if uuid is None else uuid
        self.x = x
        self.y = y
        self.done = False


class TaskQueue(object):
    def __init__(self) -> None:
        super().__init__()
        self._dict = dict()
        self._queue = Queue(1000)

    def find(self, uuid) -> Task:
        return self._dict.get(uuid, None)

    def put(self, task: Task):
        self._dict[task.uuid] = task
        self._queue.put(task)

    def get(self) -> Task:
        return self._queue.get()
