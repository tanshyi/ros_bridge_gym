import time
import random
import math
import numpy as np
import gym
from gym import spaces

from .ros_bridge import BridgeNode


class GymLab(gym.Env):

    def __init__(self, node) -> None:
        super(GymLab, self).__init__()
        self._node = node
        self.action_space = node.gym_action_space()
        self.observation_space = node.gym_observation_space()

    def reset(self):
        return self._node.gym_reset(robot=(0.,0.))

    def step(self, action):
        return self._node.gym_step(action)


class GymLabNode(BridgeNode):

    def __init__(self, name='gymlab', 
            verbose=False,
            targets=[(1.,0.)],
            reset_time=5, 
            step_time=0.2, 
            angle_limit=1.0, 
            speed_limit=(0.05,0.35), 
            range_limit=0.25, 
            norm_dist_limit=2.,
            action_in_state=True,
            max_step_per_episode=500
    ):
        super().__init__(name=name)
        self._rate = self.create_rate(100)
        self._verbose = verbose
        self._status = None
        self._epi_step = 0

        self.targets = targets
        self.reset_time = reset_time
        self.step_time = step_time
        self.angle_limit = angle_limit
        self.speed_limit = speed_limit
        self.speed_limit_mean = np.array(speed_limit).mean()
        self.range_limit = range_limit
        self.norm_dist_limit = norm_dist_limit
        self.action_in_state = action_in_state
        self.max_step_per_episode = max_step_per_episode


    @property
    def episode_status(self):
        return self._status

    @property
    def episode_steps(self):
        return self._epi_step


    def sleep(self, seconds):
        #begin = time.time()
        #while time.time() - begin < seconds:
            #self._rate.sleep()
        time.sleep(seconds)

    
    def gym_action_space(self):
        low = np.float32([-1.,-1.])
        high = np.float32([1.,1.])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def gym_observation_space(self):
        sl, sh = self.speed_limit
        low = np.float32([-self.angle_limit,float(sl),-3.15,0.] + [0.] * 24)
        high = np.float32([self.angle_limit,float(sh),3.15,self.norm_dist_limit] + [1.] * 24)
        if self.action_in_state:
            return spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            return spaces.Box(low=low[2:], high=high[2:], dtype=np.float32)


    def gym_reset(self, robot=None, target=None):
        self._status = None
        self._epi_step = 0

        if target is None:
            self._target = random.choice(self.targets)
        else:
            self._target = target
        cmd = dict(
            command = 'reset',
            target = dict(x = self._target[0], y = self._target[1])
        )
        if robot is not None:
            cmd['robot'] = dict(x = robot[0], y = robot[1]) 
            
        self.send_command(cmd)
        self.sleep(self.reset_time)

        self._prev_state = self.state.copy()
        if self.action_in_state:
            return self.observe()
        else:
            return np.float32(self.observe()[2:])


    def gym_step(self, action):
        vel_az = float(action[0]) * self.angle_limit

        sl, sh = self.speed_limit
        vel_lx = ((float(action[1]) + 1) / 2) * (sh-sl) + sl

        self._prev_state = self.state.copy()
        self.action = [vel_az, vel_lx]
        self.sleep(self.step_time)

        obs = self.observe()
        reward, reward_log, done, done_msg = self.reward_done(obs)

        self._epi_step += 1
        if self._epi_step >= self.max_step_per_episode and not done:
            self._status = "timeout"
            done = True
            done_msg = "DONE: Episode Max Step"
        
        if self._verbose:
            self.get_logger().info(f'[{self._epi_step}] {reward_log}')
            if done:
                self.get_logger().info(done_msg)

        if self.action_in_state:
            return obs, reward, done, dict()
        else:
            return np.float32(obs[2:]), reward, done, dict()

        
    def observe(self):
        state = self.state.copy()
        pos_x = float(state[0])
        pos_y = float(state[1])
        bot_angle = float(state[3])
        vel_lx = float(state[4])
        vel_az = float(state[5])
        lazer_scans = (state[6:] / 3.5).tolist()

        target_x, target_y = self._target
        target_dist = math.sqrt((target_x - pos_x)**2 + (target_y - pos_y)**2)
        target_angle = math.atan2(target_y - pos_y, target_x - pos_x)

        # calculate direction difference
        if bot_angle < 0:
            bot_angle += 2*math.pi
        if target_angle < 0:
            target_angle += 2*math.pi
        
        angle_diff = target_angle - bot_angle
        if angle_diff < -math.pi:
            angle_diff += 2*math.pi
        elif angle_diff > math.pi:
            angle_diff -= 2*math.pi

        # calculate normalised distance
        norm_dist = target_dist / math.sqrt(target_x**2 + target_y**2)

        return np.float32([vel_az, vel_lx, angle_diff, norm_dist] + lazer_scans)


    def reward_done(self, obs):
        done = False
        done_msg = None

        vel_az = float(obs[0])
        vel_lx = float(obs[1])
        angle_diff = float(obs[2])
        norm_dist = float(obs[3])
        lazer_scans = obs[4:]

        target_x, target_y = self._target
        target_dist = norm_dist * math.sqrt(target_x**2 + target_y**2)

        prev_pos_x = float(self._prev_state[0])
        prev_pos_y = float(self._prev_state[1])
        prev_target_dist = math.sqrt((target_x - prev_pos_x)**2 + (target_y - prev_pos_y)**2)
        
        r_angle = (max(0.0, (0.1 - abs(angle_diff) / math.pi) * 450)
                + min(0.0, (0.1 - abs(angle_diff) / math.pi) * 50))
        #r_distance = 50.0 - abs(norm_dist * 100)
        r_distance = (prev_target_dist - target_dist)*(5/self.step_time)*1.2*7

        #laser_reward = (sum(lazer_scans)/len(lazer_scans) - 1) * 20
        laser_reward = sum(lazer_scans) - len(lazer_scans)
        lazer_min = min(lazer_scans) * 3.5
        lazer_crashed = bool(lazer_min < self.range_limit)
        if lazer_crashed:
            self._status = "crashed"
            done_msg = "DONE: Crashed"
            done = True
            laser_crashed_reward = -200
        elif lazer_min < (2 * self.range_limit):
            laser_crashed_reward = -80
        else:
            laser_crashed_reward = 0
        
        r_collision = laser_reward + laser_crashed_reward
        
        if abs(vel_az) > 0.1:
            #angular_punish_reward = -100.0 * abs(vel_az)
            angular_punish_reward = -1
        else:
            #angular_punish_reward = 20 - 200.0 * abs(vel_az)
            angular_punish_reward = 0

        if vel_lx < self.speed_limit_mean:
            #linear_punish_reward = 20.0 * (vel_lx - self.speed_limit_mean)
            linear_punish_reward = -2
        else:
            linear_punish_reward = 0

        r_vel = angular_punish_reward + linear_punish_reward

        if target_dist < (3 * self.range_limit):
            self._status = "reached"
            done_msg = "DONE: Reached"
            done = True
            r_arrive = 100
        elif norm_dist > (self.norm_dist_limit - 0.1):
            self._status = "beyond"
            done_msg = "DONE: Beyond Range"
            done = True
            r_arrive = -100
        else:
            r_arrive = 0
            
        #reward = float(r_angle + r_distance + r_collision + r_vel + r_arrive)
        reward = float(r_distance + r_collision + r_vel + r_arrive)
        #reward_log = f'r:{reward:.2f} ang_r:{r_angle:.1f} dis_r:{r_distance:.1f} col_r:{r_collision:.1f} vel_r:{r_vel:.1f} arr_r:{r_arrive:.1f}'
        reward_log = f'r:{reward:.2f} dis_r:{r_distance:.1f} col_r:{r_collision:.1f} vel_r:{r_vel:.1f} arr_r:{r_arrive:.1f}'
        return reward, reward_log, done, done_msg

