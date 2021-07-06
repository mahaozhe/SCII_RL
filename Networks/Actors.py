"""
Actor networks
"""

import torch
import torch.nn as nn
from pysc2.lib import actions as ACTIONS
from utils.CommonLayers import Flatten, Dense2Conv, init_weights

MINIMAP_NUM = 11
SCREEN_NUM = 27


class PPOActorNet(torch.nn.Module):
    def __init__(self):
        super(PPOActorNet, self).__init__()

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

        self.layer_hidden = nn.Sequential(nn.Conv2d(32 * 3, 64, 3, stride=1, padding=1),
                                          nn.ReLU())

        self.layer_action = nn.Sequential(nn.Conv2d(64, 1, 1),
                                          nn.ReLU(),
                                          Flatten(),
                                          nn.Linear(64 * 64, len(ACTIONS.FUNCTIONS)))

        self.layer_coordinate1 = nn.Conv2d(64, 1, 1)
        self.layer_coordinate2 = nn.Conv2d(64, 1, 1)

        self.apply(init_weights)
        self.train()

    def forward(self, obs):
        obs_minimap = obs['minimap']
        obs_screen = obs['screen']
        obs_non_spatial = obs['non_spatial']

        # process observations
        m = self.minimap_conv(obs_minimap)
        s = self.screen_conv(obs_screen)
        n = self.non_spatial_dense(obs_non_spatial)

        state_h = torch.cat([m, s, n], dim=1)
        state_h = self.layer_hidden(state_h)

        pol_function_id = self.layer_action(state_h)
        pol_coordinate1 = self.layer_coordinate1(state_h)
        pol_coordinate2 = self.layer_coordinate2(state_h)

        return {'function_id': pol_function_id, 'coordinate1': pol_coordinate1, 'coordinate2': pol_coordinate2}
