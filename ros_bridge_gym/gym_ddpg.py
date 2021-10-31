import threading
import numpy as np

from stable_baselines3 import DDPG

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .ros_gym import GymLab, GymLabNode
from .noise import RandomActionNoise


class GymDDPG(GymLabNode):

    def __init__(self):
        super().__init__(name='gymlab_ddpg', action_in_state=True)
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
        #action_noise = EpsilonNormalActionNoise(mean=np.zeros(n_actions), sigma=np.ones(n_actions))
        action_noise = RandomActionNoise((n_actions,), scale=0.5)

        model = DDPG(
            policy="MlpPolicy", 
            env=env,
            action_noise=action_noise,
            train_freq=(10, 'step'),
            learning_rate=lambda x: x * 0.0002,
            gamma=0.9,
            tau=0.01,
            learning_starts=1000,
            batch_size=512,
            verbose=1
        )
        model.learn(total_timesteps=100000, log_interval=10)


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
