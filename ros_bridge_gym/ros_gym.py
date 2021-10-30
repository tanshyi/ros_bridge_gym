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

    def __init__(self, name='gymlab', 
            reset_time=5, 
            step_time=0.2, 
            angle_limit=0.3, 
            speed_limit=(0.1,0.3), 
            range_limit=0.25, 
            norm_dist_limit=2.,
            action_in_state=False
    ):
        super().__init__(name=name)
        self._rate = self.create_rate(100)

        self.reset_time = reset_time
        self.step_time = step_time
        self.angle_limit = angle_limit
        self.speed_limit = speed_limit
        self.speed_limit_mean = np.array(speed_limit).mean()
        self.range_limit = range_limit
        self.norm_dist_limit = norm_dist_limit
        self.action_in_state = action_in_state


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
        high = np.float32([self.angle_limit,float(sh),3.15,self.norm_dist_limit] + [3.51] * 24)
        if self.action_in_state:
            return spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            return spaces.Box(low=low[2:], high=high[2:], dtype=np.float32)


    def gym_reset(self):
        self._target = (4.,0.)
        cmd = dict(
            command = 'reset',
            target_x = self._target[0],
            target_y = self._target[1]
        )
        self.send_command(cmd)
        self.sleep(self.reset_time)
        if self.action_in_state:
            return self.observe()
        else:
            return np.float32(self.observe()[2:])


    def gym_step(self, action):
        vel_az = float(action[0]) * self.angle_limit

        sl, sh = self.speed_limit
        vel_lx = ((float(action[1]) + 1) / 2) * (sh-sl) + sl

        self.action = [vel_az, vel_lx]
        self.sleep(self.step_time)

        obs = self.observe()
        reward, done = self.reward_done(obs)
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

        return np.float32([vel_az, vel_lx, angle_diff, norm_dist] + lazer_scans)


    def reward_done(self, obs):
        done = False

        vel_az = float(obs[0])
        vel_lx = float(obs[1])
        angle_diff = float(obs[2])
        norm_dist = float(obs[3])
        lazer_scans = obs[4:]

        target_x, target_y = self._target
        target_dist = norm_dist * math.sqrt(target_x**2 + target_y**2)
        
        r_angle = max(0.0, (0.1 - abs(angle_diff) / math.pi) * 450)
        r_distance = 0.0 - abs(norm_dist * 100)

        #laser_reward = (sum(lazer_scans)/len(lazer_scans) - 1.5) * 20
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
        
        #r_collision = laser_reward + laser_crashed_reward
        r_collision = laser_crashed_reward
        
        if abs(vel_az) > 0.05:
            angular_punish_reward = -100.0 * abs(vel_az)
        else:
            angular_punish_reward = 20 - 200.0 * abs(vel_az)

        if vel_lx < self.speed_limit_mean:
            linear_punish_reward = 20.0 * (vel_lx - self.speed_limit_mean)
        else:
            linear_punish_reward = 0

        r_vel = angular_punish_reward + linear_punish_reward

        if target_dist < (3 * self.range_limit):
            self.get_logger().info("DONE: Reached")
            done = True
            r_arrive = 100
        elif norm_dist > (self.norm_dist_limit - 0.1):
            self.get_logger().info("DONE: Beyond Range")
            done = True
            r_arrive = -100
        else:
            r_arrive = 0
            
        #reward = float(r_angle + r_distance + r_collision + r_vel + r_arrive)
        reward = float(r_angle + r_vel)
        self.get_logger().info(f'r:{reward:.2f} ang_r:{r_angle:.1f} dis_r:{r_distance:.1f} col_r:{r_collision:.1f} vel_r:{r_vel:.1f} arr_r:{r_arrive:.1f}')
        return reward, done

