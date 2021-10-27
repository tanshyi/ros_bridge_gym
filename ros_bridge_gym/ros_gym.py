import time
import math
import numpy as np
import gym
from gym import spaces
from numpy.lib.function_base import angle

from .ros_bridge import BridgeNode


class GymLab(gym.Env):

    def __init__(self, node) -> None:
        super(GymLab, self).__init__()
        self._node = node
        self.action_space = node.gym_action_space()
        self.observation_space = node.gym_observation_space()

    def reset(self):
        return self._node.gym_reset()

    def step(self, action):
        return self._node.gym_step(action)


class GymLabNode(BridgeNode):

    def __init__(self, name='gymlab', step_time=0.1, speed_limit=(0,0.3), range_limit=0.25, norm_dist_limit=2.):
        super().__init__(name=name)
        self._rate = self.create_rate(100)

        self.step_time = step_time
        self.speed_limit = speed_limit
        self.range_limit = range_limit
        self.norm_dist_limit = norm_dist_limit


    def sleep(self, seconds):
        #begin = time.time()
        #while time.time() - begin < seconds:
            #self._rate.sleep()
        time.sleep(seconds)

    
    def gym_action_space(self):
        low = (-1.,0.)
        high = (1.,1.)
        return spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

    def gym_observation_space(self):
        low = [-1.,0.,-3.15,0.] + [0.] * 24
        high = [1.,1.,3.15,self.norm_dist_limit] + [3.51] * 24
        return spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)


    def gym_reset(self):
        self._target = (4.,0.)
        cmd = dict(
            command = 'reset',
            target_x = self._target[0],
            target_y = self._target[1]
        )
        self.send_command(cmd)
        self.sleep(3)
        return self.observe()


    def gym_step(self, action):
        vel_az = action[0]

        sl, sh = self.speed_limit
        vel_lx = action[1] * (sh-sl) + sl

        self.action = [vel_az, vel_lx]
        self.sleep(self.step_time)

        obs = self.observe()
        reward, done = self.reward_done(obs)
        return obs, reward, done, dict()

        
    def observe(self):
        state = self.state.copy()
        pos_x = float(state[0])
        pos_y = float(state[1])
        bot_angle = float(state[3])
        vel_lx = float(state[4])
        vel_az = float(state[5])
        lazer_scans = state[6:].tolist()

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

        return [vel_az, vel_lx, angle_diff, norm_dist] + lazer_scans


    def reward_done(self, obs):
        done = False

        vel_az = obs[0]
        vel_lx = obs[1]
        norm_dist = obs[3]
        lazer_scans = obs[4:]

        target_x, target_y = self._target
        target_dist = norm_dist * math.sqrt(target_x**2 + target_y**2)
        
        distance_reward = 0.0 - abs(norm_dist * 40)

        laser_reward = (sum(lazer_scans)/len(lazer_scans) - 1.5) * 20
        lazer_min = min(lazer_scans)
        lazer_crashed = bool(lazer_min < self.range_limit)
        if lazer_crashed:
            self.get_logger().info("DONE: Crashed")
            done = True
            laser_crashed_reward = -200
        elif lazer_min < (2 * self.range_limit):
            laser_crashed_reward = -80
        else:
            laser_crashed_reward = 0
        
        collision_reward = laser_reward + laser_crashed_reward
        
        if abs(vel_az) > 0.8:
            angular_punish_reward = -10
        else:
            angular_punish_reward = 0

        if vel_lx < 0.2:
            linear_punish_reward = -2
        else:
            linear_punish_reward = 0

        if target_dist < (3 * self.range_limit):
            self.get_logger().info("DONE: Reached")
            done = True
            arrive_reward = 100
        elif norm_dist > (self.norm_dist_limit - 0.1):
            self.get_logger().info("DONE: Beyond Range")
            done = True
            arrive_reward = -100
        else:
            arrive_reward = 0
            
        reward = float(distance_reward + arrive_reward + collision_reward + angular_punish_reward + linear_punish_reward)
        self.get_logger().info(f'r:{reward:.2f} dis_r:{distance_reward:.2f} col_r:{collision_reward:.2f} arr_r:{arrive_reward:.2f}')
        return reward, done

