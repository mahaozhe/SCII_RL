"""
Critic Networks
"""

import torch
import torch.nn as nn
from pysc2.lib import actions as ACTIONS
from utils.CommonLayers import Flatten, Dense2Conv, init_weights

MINIMAP_NUM = 11
SCREEN_NUM = 27


class PPOCriticNet(torch.nn.Module):
    def __init__(self):
        super(PPOCriticNet, self).__init__()

        self.minimap_conv = nn.Sequential(nn.Conv2d(MINIMAP_NUM, 16, 5, stride=1, padding=2),
                                          nn.ReLU(),
                                          nn.Conv2d(16, 32, 3, stride=1, padding=1),
                                          nn.ReLU())
        self.screen_conv = nn.Sequential(nn.Conv2d(SCREEN_NUM, 16, 5, stride=1, padding=2),
                                         nn.ReLU(),
                                         nn.Conv2d(16, 32, 3, stride=1, padding=1),
                                         nn.ReLU())

        self.non_spatial_dense = nn.Sequential(nn.Linear(len(ACTIONS.FUNCTIONS), 32),
                                               nn.ReLU(),
                                               Dense2Conv())

        self.layer_hidden = nn.Sequential(nn.Conv2d(32 * 5, 64, 3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(64, 1, 1),
                                          nn.ReLU(),
                                          Flatten())

        self.layer_value = nn.Linear(64 * 64, 1)

        self.apply(init_weights)
        self.train()

    def forward(self, obs):
        """
        The forward function for a critic in PPO algorithm returns the value given one obs
        :param obs: the current observation
        :return: the estimated value for the given observation
        """

        obs_minimap = obs['minimap']
        obs_screen = obs['screen']
        obs_non_spatial = obs['non_spatial']

        m = self.minimap_conv(obs_minimap)
        s = self.screen_conv(obs_screen)
        n = self.non_spatial_dense(obs_non_spatial)

        sh = torch.cat([m, s, n], dim=1)
        sh = self.layer_hidden(sh)
        v = self.layer_value(sh)
        return v
