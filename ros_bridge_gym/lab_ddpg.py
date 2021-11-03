import threading

from stable_baselines3 import DDPG

import rclpy

from .ros_gym import GymLabNode
from .lab_node import Task, TaskQueue


eval_targets = [
    (4.,-1.),
    (7.,1.),
    (9.,-2.),
    (0.,7.),
    (-3.,1.),
    (-4.,-1.),
    (-8.,1.)
]

class LabDDPG(GymLabNode):

    def __init__(self):
        super().__init__(name='lab_ddpg')
        self._timer = self.create_timer(1, self._timer_callback)
        self._ready = threading.Event()
        self._queue = TaskQueue()

    def _timer_callback(self):
        self._ready.set()


    def eval(self, targets, model_path='model/ddpg'):
        for t in targets:
            self._queue.put(Task(x=t[0], y=t[1]))

        m = DDPG.load(model_path)

        self._ready.wait(timeout=None)
        while True:
            task = self._queue.get()
            self.targets = [(task.x, task.y)]

            done = False
            obs = self.gym_reset()
            while not done:
                action, _ = m.predict(obs)
                obs, reward, done, _ = self.gym_step(action)

            task.done = True
            self.get_logger().info(f'task done: [{self._status}]')


def main(args=None):
    rclpy.init(args=args)

    node = LabDDPG()
    try:
        threading.Thread(target=lambda:node.eval(eval_targets)).start()
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
