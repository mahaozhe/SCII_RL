"""
To implement classes for replay buffer.
"""

import numpy as np
import scipy.signal

from collections import namedtuple, deque
import random


class BaseReplayBuffer:
    """
    The base class for all replay buffers.
    """

    def __init__(self, *args):
        """
        initialization.
        """
        pass

    def __len__(self):
        """
        return the length of the replay buffer.
        :return: the length
        """
        raise NotImplementedError

    def store(self, *args):
        """
        Store transitions/trajectories into the replay buffer
        """
        raise NotImplementedError

    def sample(self, *args):
        """
        Sample a batch of transitions/trajectories from the replay buffer
        :return: a batch of samples
        """
        raise NotImplementedError


class PPOReplayBuffer(BaseReplayBuffer):
    """
    A replay buffer for PPO algorithm to store trajectories and using Generalized Advantage Estimation (GAE-Lambda) for
    calculating the advantages of state-actions pairs.
    """

    def __init__(self, gamma, lamb):
        """
        initialization.
        """
        super(PPOReplayBuffer, self).__init__()

        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.return_buffer = []
        self.value_buffer = []
        self.advantage_buffer = []
        self.log_prob_buffer = []

        self.gamma = gamma
        self.lamb = lamb

        self.start = 0
        self.end = 0

    def __len__(self):
        return self.end

    def store(self, obs_np, action_np, reward, value, log_prob):
        """
        Store the transition of one time-step

        :param obs_np: a dict with 'screen', 'minimap' and 'non_spatial' keys
        :param action_np: a dict with `function_id`, `coordinate1` and `coordinate2` keys
        :param reward: the reward
        :param value: the value
        :param log_prob: the log probabilities for all function_ids
        """

        self.obs_buffer.append(obs_np)
        self.action_buffer.append(action_np)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)
        self.log_prob_buffer.append(log_prob)

        self.end += 1

    def sample(self):
        """
        To return the trajectories in the whole epoch and clear the buffer, leave the data-type-normalization
        to the PPO algorithm class.

        :return: the transitions data: (obs_np_all, action_np_all, return_all, advantage_all, log_prob_all)
        """

        assert len(self.obs_buffer) == len(self.action_buffer) == len(self.return_buffer) == len(
            self.advantage_buffer) == len(self.log_prob_buffer), "The sub-buffers are not with the same length"

        transitions = {'obs_nps': self.obs_buffer, 'action_nps': self.action_buffer, 'returns': self.return_buffer,
                       'advantages': self.advantage_buffer, 'log_probs': self.log_prob_buffer}

        return transitions

    def clear(self):
        del self.obs_buffer[:]
        del self.action_buffer[:]
        del self.reward_buffer[:]
        del self.return_buffer[:]
        del self.value_buffer[:]
        del self.advantage_buffer[:]
        del self.log_prob_buffer[:]

        self.start = 0
        self.end = 0

    def complete_trajectory(self, last_value):
        """
        The function to complete the trajectory, compute the returns, advantages along the trajectory

        :param last_value: the last value, if the trajectory is cut off by the trajectory-time-step limit or
        the ending of the epoch, value should be bootstrapped by Critic(s_T), otherwise, value should be 0.
        """

        trajectory_slice = slice(self.start, self.end)
        rewards = np.array(self.reward_buffer[trajectory_slice] + [last_value])
        values = np.array(self.value_buffer[trajectory_slice] + [last_value])

        # ! implement the GAE advantage calculation (need numpy array)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer[trajectory_slice] = self._cumulate_discount(deltas, self.gamma * self.lamb).tolist()
        self.return_buffer[trajectory_slice] = self._cumulate_discount(rewards, self.gamma)[:-1].tolist()

        self.start = self.end

    def _cumulate_discount(self, x, discount):
        """
        compute the cumulative discounted value for x

        :param x: the numpy array to compute
        :param discount: the discount factor
        :return: the cumulative discount
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


DDPGTransition = namedtuple('Transition', ('obs', 'action', 'obs_next', 'reward', 'done'))


class DDPGReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def store(self, *args):
        """
        Append a sample
        :param args: a sample in order of (obs, action, obs_next, reward, done)
        """

        self.buffer.append(DDPGTransition(*args))

    def sample(self, batch_size):
        """
        To sample a batch of samples

        :param batch_size: batch size
        :return: a batch of sample
        """

        return random.sample(self.buffer, batch_size)
