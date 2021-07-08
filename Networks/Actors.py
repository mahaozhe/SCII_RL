"""
Actor networks
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
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

    def forward(self, obs, action=None):
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

        # TODO: change the log_prob

        if action is not None:
            probable_function_id = nn.Softmax(dim=-1)(action['function_id'])
            probable_function_id = probable_function_id * obs_non_spatial

            distribution = Categorical(probable_function_id)

            function_id = distribution.sample()
            log_prob = {'function_id': distribution.log_prob(function_id)}

            batch_size = action['function_id'].size()[0]

            coordinate_position1 = nn.Softmax(dim=-1)(action['coordinate1'].view(batch_size, -1))
            coordinate_position1_distribution = Categorical(coordinate_position1)
            coordinate_position1 = coordinate_position1_distribution.sample()
            log_prob['coordinate1'] = coordinate_position1_distribution.log_prob(coordinate_position1)

            coordinate_position2 = nn.Softmax(dim=-1)(action['coordinate2'].view(batch_size, -1))
            coordinate_position2_distribution = Categorical(coordinate_position2)
            coordinate_position2 = coordinate_position2_distribution.sample()
            log_prob['coordinate2'] = coordinate_position2_distribution.log_prob(coordinate_position2)

        else:
            log_prob = None

        return {'function_id': pol_function_id, 'coordinate1': pol_coordinate1,
                'coordinate2': pol_coordinate2}, log_prob
