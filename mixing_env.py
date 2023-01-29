import gym
from gym import spaces
import random
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

"""
This RL environment was created based on the Continuous Product Blending Simulation given here:

https://jckantor.github.io/CBE30338/02.04-Continuous-Product-Blending.html

"""


def deriv(X, t, qa, qs, qout, CAF):
    """
    Function to calculate derivatives of volume and concentration

    Args:
        X - list of values for the model state
        t - variable containing value corresponding to current time
        qa - flow rate of feed tank
        qs - flow rate of solvent
        qout - flow rate out of mixer
        CAF - concentration in feed flow rate

    Returns:
        dV - derivative of volume
        dca - derivative of concentration
    """
    V, ca = X
    dV = qa + qs - qout
    dca = qa * (CAF - ca) / V - qs * ca / V
    return dV, dca


class MixEnv(gym.Env):
    """
    Class defining the mixer environment
    """

    def __init__(self):
        # initialize
        super(MixEnv, self).__init__()
        # defining some constants
        self.CAF = 200  # g/liter
        self.VMAX = 15000  # liters
        self.VMIN = 8000  # liters
        self.CA_SP = 8  # g/liter
        self.RTIME_SP = 96  # residence time in hours
        self.QOUT_BAR = 125  # nominal flowrate out of mixer
        self.DT = 1  # timestep
        # defining some limits for actions
        self.QA_MIN = 0  # minimum for feed flowrate
        self.QA_MAX = 10  # max for feed flowrate
        self.QS_MIN = 0  # minimum for solvent flowrate
        self.QS_MAX = 200  # max for solvent flowrate
        # defining some other limits for termination of episode
        self.CA_MIN = 0  # minimum for outlet concentration
        self.CA_MAX = 20  # maximum for outlet concentration
        self.RTIME_MIN = 0  # minimum for residence time
        self.RTIME_MAX = 300  # max for residence time
        # nominal steady state process values
        self.v_bar = self.RTIME_SP * self.QOUT_BAR
        self.ca_bar = self.CA_SP
        self.qa_bar = self.CA_SP * self.QOUT_BAR / self.CAF
        self.qs_bar = self.QOUT_BAR - self.qa_bar
        # defining the action space - normalized for ease of training
        self.action_space = spaces.Box(
            low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,), dtype=np.float32
        )
        # defining the observation space - normalized for ease of training
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            shape=(3,),
            dtype=np.float32,
        )

    def calc_action(self, norm_action, min, max):
        return min + ((norm_action + 1) / 2) * (max - min)

    def normalize_states(self, state, min, max):
        return 2 * ((state - min) / (max - min)) - 1

    def step(self, action):
        # step through an iteration
        truncated = False
        terminated = False
        # convert returned action to actual values
        self.qa = self.calc_action(action[0], self.QA_MIN, self.QA_MAX)
        self.qs = self.calc_action(action[1], self.QS_MIN, self.QS_MAX)
        # use the action on the environment and simulate next time step
        self.volume, self.ca = odeint(
            deriv,
            [self.volume, self.ca],
            [self.t, self.t + self.DT],
            (self.qa, self.qs, self.qout, self.CAF),
        )[-1]
        # calculate residence time
        self.rtime = self.volume / self.qout
        # update simulation time
        self.t += self.DT
        # update the time, states and actions to the history buffer
        self.history.append(
            [self.t, self.volume, self.ca, self.rtime, self.qs, self.qa]
        )
        # the episode truncation limit is set at 500 s
        if self.t > 500:
            truncated = True
        # check for terminal conditions. anything unphysical or unrealistic
        if self.volume < 0:
            # terminate if calculated volume is negative
            terminated = True
        elif self.volume > self.VMAX * 1.5:
            # terminate if volume is too high - try again
            terminated = True

        if self.ca < 0:
            # terminate if calculated outlet concentration is negative
            terminated = True

        if self.rtime < 10:
            # terminate if residence time is too low
            terminated = True
        # translate the observation states into -1 to +1 range
        observation = [
            np.float32(self.normalize_states(self.volume, self.VMIN, self.VMAX)),
            np.float32(self.normalize_states(self.ca, self.CA_MIN, self.CA_MAX)),
            np.float32(
                self.normalize_states(self.rtime, self.RTIME_MIN, self.RTIME_MAX)
            ),
        ]
        observation = np.array(observation)
        # return an empty info dict
        info = {}
        # calculate reward
        # concentration_reward
        conc_distance = abs(self.CA_SP - self.ca)
        conc_reward = 1 - (conc_distance / 10) ** 2  # shaped to get max reward at 8 g/l
        # volume_reward
        vol_reward = (
            1.0 if (self.volume > 8000) & (self.volume < 15000) else -10
        )  # as long as volume limits are not terminated +1 else -10
        # residencetime_reward
        rtime_distance = abs(
            self.RTIME_SP - self.volume / self.qout
        )  # shaped to get max reward at 96 hours
        rtime_reward = 1 - (rtime_distance / 30) ** 2
        # total reward is the summation of these individual rewards
        reward = conc_reward + vol_reward + rtime_reward

        return observation, reward, terminated, truncated, info

    def reset(self):
        # resets the environment
        info = {}
        self.t = 0  # initiate timestep to 0
        # initiate qa and qs to steady state process values
        self.qa = self.qa_bar
        self.qs = self.qs_bar
        # initiate volume to steady state values
        self.volume = self.v_bar
        # randomize concentration at start of episode
        self.ca = random.uniform(0, 16)
        # randomize outlet flowrate
        self.qout = random.uniform(0.9, 1.1) * self.QOUT_BAR
        self.rtime = self.volume / self.qout
        # get observations
        observation = [
            np.float32(self.normalize_states(self.volume, self.VMIN, self.VMAX)),
            np.float32(self.normalize_states(self.ca, self.CA_MIN, self.CA_MAX)),
            np.float32(
                self.normalize_states(self.rtime, self.RTIME_MIN, self.RTIME_MAX)
            ),
        ]
        observation = np.array(observation)
        # initiate a history buffer
        self.history = [[self.t, self.volume, self.ca, self.rtime, self.qs, self.qa]]
        return observation, info

    def visualize(self, labels):
        # plots a simulation history
        history = np.array(self.history)
        t = history[:, 0]
        n = len(labels) - 1
        plt.figure(figsize=(8, 1.95 * n))
        plt.suptitle(f"Outlet rate = {str(round(self.qout, 2))}")
        for k in range(0, n):
            plt.subplot(n, 1, k + 1)
            plt.plot(t, history[:, k + 1])
            plt.title(labels[k + 1])
            plt.xlabel(labels[0])
            plt.grid()
        plt.tight_layout()
        plt.show()
