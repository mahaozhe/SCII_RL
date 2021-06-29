"""
A template to use the pysc2 environment.

Agent: using the random agent, referring to: pysc2.agent.random_agent.RandomAgent
Environment: using the 'Simple64' environment, both the player and the build-in AI are the race of terran.

You can simply run the agent in the environment by the command:
    python pysc2EnvTemplate.py

Our you can use the `pysc2.bin.agent` module to run the MyAgent individually:
    python -m pysc2.bin.agent --map <MAP_NAME> --agent pysc2EnvTemplate.MyAgent
"""

import numpy as np

# import the BaseAgent class which we should derive from
from pysc2.agents import base_agent
# import actions
from pysc2.lib import actions
# import features
from pysc2.lib.features import AgentInterfaceFormat, Dimensions
# import the SC2Env environment
from pysc2.env.sc2_env import SC2Env, Agent, Bot, Race, Difficulty
# import absl
from absl import app


# define the agent, overriding the `step` function
class MyAgent(base_agent.BaseAgent):
    """
    My customized agent, simply copied from `pysc2.agents.random_agent.RandomAgent`.
    """

    def step(self, obs):
        super(MyAgent, self).step(obs)

        function_id = np.random.choice(obs.observation.available_actions)

        args = [[np.random.randint(0, size) for size in arg.sizes] for arg in
                self.action_spec.functions[function_id].args]

        return actions.FunctionCall(function_id, args)


# define a loop function for only one agent
def loop(agent, env, max_steps=200):
    # setup the agent
    # ! Important: here, to setup the agent, the `env.observation_spec()` and `env.action_spec()` return tuples,
    # ! however, we want the elements inside, so we need the indexes `[0]` form them respectively.
    observation_spec = env.observation_spec()[0]
    action_spec = env.action_spec()[0]
    agent.setup(observation_spec, action_spec)

    # reset the environment and the agent
    # ! Note: the `env.reset()` returns a tuple,
    # ? each element for each agent/bot.
    timesteps = env.reset()
    agent.reset()

    total_steps = 0

    while True:
        # ! Note: the `env.step()` function needs a list, which actually allows multiple actions per frame,
        # ! so, we need to pack up the single action returned by the `agent.step()` function.
        # ! However, if the `agent.step()` function returns a list itself, then there is no need to use the list.
        action = [agent.step(timesteps[0])]

        if timesteps[0].last():
            break

        total_steps += 1
        if total_steps > max_steps:
            break

        timesteps = env.step(action)


# define the main function
def main(args):
    agent = MyAgent()

    # define the environment
    env = SC2Env(map_name="MoveToBeacon",
                 players=[Agent(Race.terran)],
                 agent_interface_format=AgentInterfaceFormat(feature_dimensions=Dimensions(screen=64, minimap=64)),
                 step_mul=16, visualize=True)

    try:
        loop(agent, env)
    except KeyboardInterrupt:
        pass
    finally:
        print("Finished!")


if __name__ == "__main__":
    app.run(main)
