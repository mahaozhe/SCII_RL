"""
The main script to run the training or evaluation
"""

from absl import app, logging
from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Dimensions, Agent, Race

from pysc2.lib import actions

import numpy as np

from Algorithms.PPO import PPO

# hyper-parameters
MAP_SIZE = 64


def temp_actor(obs):
    return actions.FunctionCall(0, []), 0.5, 0.2


def main(args):
    # instantiate the environment
    agent_interface_format = AgentInterfaceFormat(feature_dimensions=Dimensions(screen=MAP_SIZE, minimap=MAP_SIZE))

    env = SC2Env(map_name="MoveToBeacon", players=[Agent(Race.terran)], agent_interface_format=agent_interface_format,
                 step_mul=16, visualize=False)

    # instantiate the agent/algorithm
    agent = PPO(env, temp_actor, temp_actor, save_epochs=10)
    agent.learn(100)


if __name__ == "__main__":
    app.run(main)
