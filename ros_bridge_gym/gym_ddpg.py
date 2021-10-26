import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .ros_gym import GymLab, GymLabNode


class GymDDPG(GymLabNode):

    def __init__(self):
        super().__init__(name='gymlab_ddpg')
        self._timer = self.create_timer(5, self._timer_callback)
        self._training = False

    def _timer_callback(self):
        self.train()


    def train(self):
        if self._training:
            return
        else:
            self._training = True
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
        rclpy.spin(node, executor=MultiThreadedExecutor(4))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
