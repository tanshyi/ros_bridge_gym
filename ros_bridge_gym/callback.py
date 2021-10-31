import os
import datetime
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, log_dir, check_freq=1000, check_episodes=5, save_replay=False, verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')

        self.check_freq = check_freq
        self.check_episodes = check_episodes
        self.save_replay = save_replay
        self.best_mean_reward = -np.inf


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)


    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        # Retrieve training reward
        x, y = ts2xy(load_results(self.log_dir), 'episodes')
        if len(x) == 0:
            return True
        
        # Mean training reward over the last 100 episodes
        mean_reward = np.mean(y[-self.check_episodes:])
        if self.verbose > 0:
            print(f"Num timesteps: {self.num_timesteps}")
            print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

        # New best model, you could save the agent here
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward

            # Example for saving best model
            now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            model_path = os.path.join(self.save_path, f'model-{now}-t{self.n_calls}-e{len(x)}')
            replay_path = f'{model_path}-replay'

            if self.verbose > 0:
                print(f"Saving new best model to {model_path}")
            self.model.save(model_path)

            if self.save_replay:
                self.model.save_replay_buffer(replay_path)

        return True

