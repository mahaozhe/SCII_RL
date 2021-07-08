"""
The main script to run the training or evaluation
"""

from absl import app, logging
from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Dimensions, Agent, Race

from Algorithms.PPO import PPO
from Networks.Actors import PPOActorNet
from Networks.Critics import PPOCriticNet

# ! hyper-parameters
# * for the environment
MAP_NAME = "MoveToBeacon"
MAP_SIZE = 64
STEP_INTERVAL = 16
VISUALIZE_FEATURE_MAPS = False

# * for the training
GAMMA = 0.99
CLIP_RATIO = 0.2
ACTOR_LEARNING_RATE = 0.0003
CRITIC_LEARNING_RATE = 0.001
ACTOR_TRAINING_ITERATIONS = 80
CRITIC_TRAINING_ITERATIONS = 80
LAMBDA = 0.97
TARGET_KL = 0.1
RANDOM_SEED = 1234

MODEL_NAME = "MTB_V1"
SAVE_EPOCHS = 100
EPOCHS = 1000

# * for restoring training
RESTORE = False
TOKEN = "final"


def main(args):
    # instantiate the environment
    agent_interface_format = AgentInterfaceFormat(feature_dimensions=Dimensions(screen=MAP_SIZE, minimap=MAP_SIZE))

    env = SC2Env(map_name=MAP_NAME,
                 players=[Agent(Race.terran)],
                 agent_interface_format=agent_interface_format,
                 step_mul=STEP_INTERVAL,
                 visualize=VISUALIZE_FEATURE_MAPS)

    # instantiate the agent/algorithm
    agent = PPO(env=env,
                actor=PPOActorNet(),
                critic=PPOCriticNet(),
                gamma=GAMMA,
                clip_ratio=CLIP_RATIO,
                actor_lr=ACTOR_LEARNING_RATE,
                critic_lr=CRITIC_LEARNING_RATE,
                actor_train_iterations=ACTOR_TRAINING_ITERATIONS,
                critic_train_iterations=CRITIC_TRAINING_ITERATIONS,
                lamb=LAMBDA,
                target_kl=TARGET_KL,
                seed=RANDOM_SEED,
                map_size=MAP_SIZE,
                model_name=MODEL_NAME,
                save_epochs=SAVE_EPOCHS)

    if RESTORE:
        agent.restore(TOKEN, EPOCHS)
    else:
        agent.learn(EPOCHS)


if __name__ == "__main__":
    app.run(main)
