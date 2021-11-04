import time
import os
from uuid import uuid4
from queue import Queue
from threading import Event, Thread

from stable_baselines3 import DDPG, TD3, SAC, PPO
from sb3_contrib import TQC

import rclpy

from .ros_gym import GymLabNode


class Task(object):
    def __init__(self, x, y, uuid=None) -> None:
        super().__init__()
        self.uuid = str(uuid4()) if uuid is None else uuid
        self.x = x
        self.y = y
        self.done = False
        self.status = None
        self.reward = 0.
        self.time = 0.
        self.steps = 0


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


class LabNode(GymLabNode):

    def __init__(self):
        super().__init__(name='lab', reset_time=3)
        self._timer = self.create_timer(1, self._timer_callback)
        self._ready = Event()
        self._queue = TaskQueue()
        self.declare_parameter('algo', 'tqc')

    def _timer_callback(self):
        self._ready.set()


    @property
    def algorithm(self):
        return self.get_parameter('algo').value

    
    def load_model(self, model_path=None):
        algo = self.algorithm
        if model_path is None:
            model_path = os.path.join('model', algo)

        if algo == 'ddpg':
            return DDPG.load(model_path)
        elif algo == 'td3':
            return TD3.load(model_path)
        elif algo == 'sac':
            return SAC.load(model_path)
        elif algo == 'tqc':
            return TQC.load(model_path)
        elif algo == 'ppo':
            return PPO.load(model_path)
        else:
            raise RuntimeError()


    def eval(self, targets=None, model_path=None):
        if targets is not None:
            for t in targets:
                self._queue.put(Task(x=t[0], y=t[1]))

        self.get_logger().info(f'loading model: [{self.algorithm}]')
        m = self.load_model(model_path)

        self._ready.wait(timeout=None)
        self.gym_reset(robot=(0.,0.))

        while True:
            task = self._queue.get()
            self.get_logger().info(f'task begin: [{task.x:.1f}, {task.y:.1f}]')
            time_begin = time.time()

            obs = self.gym_reset(target=(task.x, task.y))
            while not task.done:
                action, _ = m.predict(obs)
                obs, reward, task.done, _ = self.gym_step(action)
                task.reward += reward
                task.steps += 1

            task.time = time.time() - time_begin
            task.status = self.episode_status
            if task.status == 'reached':
                self.gym_reset(target=(task.x, task.y))
            else:
                self.gym_reset(target=(task.x, task.y), robot=(0.,0.))
            
            self.get_logger().info(f'task done: [{task.status}] steps: {task.steps} time: {task.time:.1f} reward: {task.reward:.1f}')


eval_targets = [
    (9.,-1.),
    (7.,1.),
    (4.,-1.),
    (0.,4.),
    (-3.,1.),
    (-4.,0.),
    (-8.,1.)
]

def main(args=None):
    rclpy.init(args=args)

    node = LabNode()
    try:
        Thread(target=lambda:node.eval(eval_targets)).start()
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
