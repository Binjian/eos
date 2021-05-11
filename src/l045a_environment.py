import random
from typing import List
import numpy as np

Vectorf = List[float]
Vectori = List[int]

"""
# vanilla discrete environment : Fully Observable MDP! 

## Observation: 
 - velocity $v(t)$
 - mileage $d(t)$
 
## Action:
 - acceleration $a_k$ = 1, 0, -1
 - unit acceleration set to 5 km/h
 
## Reward:
 - $r_t = |a|^2 ~ I^2 ~ P$
 - $G = \sigma_t^{r_t} = w $ energy consumption, objective is to minimize the long-term reward the overall energy consumption
 - end point punishment: end point velocity $v_E$ will need $K=\frac{v_E}{5}$ periods to brake to stand still
    - $r_E=\sigma_{k=0}^{K} k^2$
    
## Transition Probability, given action $a(t)$:
 - deterministic
 - period: 1 hour
 - $v_{(k+1)T} = v(kT) + a_k (t-kT)$ $v((k+1)T)=v(kT)+a_k\cdot T$
 - if $v_{(k+1)}<0$ then $v_{(k+1)}=0$  
 - $d_k = \int_{kT}^{t} (v(kT)+a_k (t-kT)dt = v(kT)\cdot T + \frac{1}{2}a_k T^2$
 - mileage $ m(t) = \sigma_{l=0}^{k} d_l 
 
## Discount factor
$\gamma=1$ no discounting, since energy consumption has no discounting effect.

"""


class Environment:
    def __init__(self):
        self.trip = 100  # 100 km
        self.powerK = 40  # proportion parameter for convert acceleration into power
        self.A = [-0.1, 0, 0.1]  # Action space, acceleration capabiltiy is 0.1m/s^2
        self.distance = 0.0  # Starting at origin0 in km
        self.velocity = 0.0  # Starting standing still in m/s
        self.acc = 0.0  # Starting standing still in m/s^2
        # self.reward = 0.0  # immediate reward acc^2 in m^2/s^4 (accumulated reward, consumed energy)
        self.gamma = 1.0  # discounting factor, 1.0 means no discounting
        # self.G = 0         # return is to be estimated,  environment just has to provide immediate reward

    def get_observation(self) -> Vectorf:
        return [self.distance, self.velocity]

    #  return action
    def get_actions(self) -> float:
        return self.acc

    def is_done(self) -> bool:
        return self.distance >= self.trip

    def reset(self):
        self.distance = 0.0  # Starting at origin0
        self.velocity = 0.0  # Starting standing still
        self.acc = 0.0  # Starting standing still

        observation = [0.0, 0.0]
        reward = 0
        done = False
        info = []
        return [observation, reward, done, info]

    #  update system dynamics, observation, reward, done (accumulated distance travelled)
    def step(self, action: int) -> Vectorf:
        # mapping action to actual acceleration
        acc = (
            -0.1 if action == 0 else (0.0 if action == 1 else 0.1)
        )  # acc = 0.1m/s^2 (0.01g) [0, 1, 2] --> [-0.1, 0.0, 0.1]

        #  update system dynamics: velocity
        vk = self.velocity
        vk1 = self.velocity + acc * 60
        #  limit velocity between [0, 16.66], with 0.1m/s^2, can reach 21.6 km/h (6m/s) within a minute with no constraints
        self.velocity = 0 if vk1 < 0 else (16.67 if vk1 >= 16.67 else vk1)

        #  update system dynamics: distance
        dist = self.distance
        if self.velocity <= 1e-3:
            self.distance = dist
        else:
            dk = (vk * 60 + acc * 60 * 60 / 2.0) / 1000  # convert from m to km
            self.distance = dist + dk if dk > 0 else dist

        observation = [self.distance, self.velocity]

        #  update reward
        power = self.powerK * np.square(np.abs(acc))
        reward = -power  # immediate reward is the minus power

        #  update done and info
        done = self.is_done()
        if done:
            ve = self.velocity
            time_unit = (ve / (0.1 * 60)) * self.powerK * np.square((np.abs(0.1)))
            terminal_reward = -time_unit
            reward += terminal_reward

        info = []

        return [observation, reward, done, info]
