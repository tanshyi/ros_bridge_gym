import time
import threading
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .ros_gym import GymLab, GymLabNode


class GymDDPG(GymLabNode):

    def __init__(self):
        super().__init__(name='gymlab_ddpg', speed_limit=(0.2,0.7))
        self._timer = self.create_timer(1, self._timer_callback)
        self._ready = threading.Event()
        self._training = False

    def _timer_callback(self):
        self._ready.set()


    def train(self):
        if self._training:
            return
        else:
            self._training = True
            self._ready.wait(timeout=None)
            self.get_logger().info("training begin")
        
        env = GymLab(node=self)

        # The noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
        model.learn(total_timesteps=10000, log_interval=10)


def main(args=None):
    rclpy.init(args=args)

    node = GymDDPG()
    try:
        threading.Thread(target=node.train).start()
        rclpy.spin(node, executor=MultiThreadedExecutor(4))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
