import threading

from stable_baselines3 import DDPG

import rclpy

from .ros_gym import GymLabNode
from .lab_node import Task, TaskQueue


eval_targets = [
    (9.,-1.),
    (7.,1.),
    (4.,-1.),
    (0.,4.),
    (-3.,1.),
    (-4.,0.),
    (-8.,1.)
]

class LabDDPG(GymLabNode):

    def __init__(self):
        super().__init__(name='lab_ddpg', reset_time=3)
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
        self.gym_reset(robot=(0.,0.))

        while True:
            task = self._queue.get()
            self.get_logger().info(f'task begin: [{task.x:.1f}, {task.y:.1f}]')

            obs = self.gym_reset(target=(task.x, task.y))
            while not task.done:
                action, _ = m.predict(obs)
                obs, reward, task.done, _ = self.gym_step(action)
                task.reward += reward
                task.steps += 1

            task.status = self.episode_status
            if task.status == 'reached':
                self.gym_reset(target=(task.x, task.y))
            else:
                self.gym_reset(target=(task.x, task.y), robot=(0.,0.))
            
            self.get_logger().info(f'task done: [{task.status}] steps: {task.steps} reward: {task.reward:.1f}')


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
