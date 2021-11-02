import os
import datetime
import threading
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .ros_gym import GymLab, GymLabNode
from .callback import ModelCheckpointCallback, SaveOnBestTrainingRewardCallback


def monitor(env, log_dir=None):
    if log_dir is None:
        home = os.path.expanduser('~')
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(home, 'Workspace', 'ros_gym', f'monitor_{now}')

    os.makedirs(log_dir, exist_ok=True)
    return Monitor(env, log_dir), log_dir


train_targets = [
    (4.,-1.),
    (7.,1.),
    (9.,-2.),
    (0.,7.),
    (-3.,1.),
    (-4.,-1.),
    (-8.,1.)
]

class GymDDPG(GymLabNode):

    def __init__(self):
        super().__init__(name='gymlab_ddpg', targets=train_targets)
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
        
        env, log_dir = monitor(GymLab(node=self))

        callback = CallbackList([
            ModelCheckpointCallback(1000, log_dir, save_replay=False),
            SaveOnBestTrainingRewardCallback(log_dir, save_replay=False)
        ])

        model = PPO(
            policy="MlpPolicy", 
            env=env,
            n_steps=128,
            n_epochs=5,
            learning_rate=lambda x: x * 0.0003,
            gamma=0.99,
            batch_size=128,
            verbose=1
        )
        model.learn(total_timesteps=1000000, log_interval=10, callback=callback)


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
