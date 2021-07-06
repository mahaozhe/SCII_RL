"""
Proximal Policy Optimization (by clipping) with early stopping based on approximate KL.

Reference:
- [PPO] Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
- [GAE] High-Dimensional Continuous Control Using Generalized Advantage Estimation: https://arxiv.org/abs/1506.02438
- OpenAI Spinning Up implemented PPO algorithm: https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""

import numpy as np
from pysc2.lib import actions as ACTIONS
import torch

import os
import copy

from utils.ReplayBuffer import PPOReplayBuffer


class PPO:
    """
    The Proximal Policy Optimization (by clipping) agent
    """

    # TODO: abstract a base class

    def __init__(self, env, actor, critic, device=None, gamma=0.99, clip_ratio=0.2, actor_lr=0.0003,
                 critic_lr=0.001, actor_train_iterations=80, critic_train_iterations=80, lamb=0.97,
                 max_trajectory_length=1000, target_kl=0.01, seed=0, action_space=len(ACTIONS.FUNCTIONS), map_size=64,
                 batch_size=32, warmup_steps=1000, save_path="./Saves/", model_name='UnnamedModel', save_epochs=100):
        """
        Initialization.

        :param env:
        :param actor:
        :param critic:
        :param replay_buffer:
        :param device:
        :param gamma:
        :param clip_ratio:
        :param actor_lr:
        :param critic_lr:
        :param actor_train_iterations:
        :param critic_train_iterations:
        :param lamb:
        :param max_trajectory_length:
        :param target_kl:
        :param seed:
        :param action_space:
        :param map_size:
        :param batch_size:
        :param warmup_steps:
        :param save_path:
        :param model_name:
        :param save_epochs:
        """

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # initialize the environment
        self.env = env

        # initialize the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # initialize the policy actor and the target actor
        self.actor = actor.to(self.device)
        # initialize the policy critic and the target critic
        self.critic = critic.to(self.device)
        # initialize two optimizers using Adam
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)

        # for the algorithm hyper-parameters
        # self.gamma = gamma
        self.clip_ratio = clip_ratio
        # self.lamb = lamb
        self.target_kl = target_kl
        self.actor_train_iterations = actor_train_iterations
        self.critic_train_iterations = critic_train_iterations
        self.max_trajectory_length = max_trajectory_length

        # initialize the replay buffer
        self.replay_buffer = PPOReplayBuffer(gamma, lamb)
        # for the batch size of training
        self.batch_size = batch_size

        # specify the action space, a number to identify the number of possible actions
        self.action_space = action_space
        # for the map size
        self.map_size = map_size

        # for training warm-up steps
        self.warmup_steps = warmup_steps
        # to record how many iterations
        self.iteration = 0

        # model and training information saved location
        self.model_name = model_name
        self.save_path = os.path.join(save_path, model_name)
        os.makedirs(self.save_path, exist_ok=True)
        self.check_point_save_epochs = save_epochs

        # to record the epoch cumulative rewards
        self.epoch_rewards = []

    def _state_2_obs_np(self, state):
        """
        The helper to transform a SC2Env-returned state to an obs dict with numpy values.

        :param state: a state from SC2Env
        :return: an obs dict for agent with numpy value
        """

        avail_actions = np.zeros((self.action_space,), dtype='float32')
        avail_actions[state.observation['available_actions']] = 1

        obs_np = {'minimap': state.observation.feature_minimap,
                  'screen': state.observation.feature_screen,
                  'non_spatial': avail_actions}

        assert obs_np['minimap'].shape == (11, 64, 64), "obs_np.minimap is in the wrong shape"
        assert obs_np['screen'].shape == (27, 64, 64), "obs_np.screen is in the wrong shape"
        assert obs_np['non_spatial'].shape == (573,), "obs_np.non_spatial is in the wrong shape"

        return obs_np

    def _function_call_2_action_np(self, function_call):
        """
        The helper to transform a SC2Env-interacted FunctionCall to an action dict with numpy values.

        :param function_call: an SC2Env-interacted FunctionCall instance
        :return: an action dict with numpy values
        """

        function_id = np.zeros(shape=(self.action_space,), dtype='float32')
        function_id[function_call.function] = 1

        coordinates = [np.zeros(shape=(1, 64, 64), dtype='float32')] * 2

        c_i = 0
        for arg in function_call.arguments:
            if arg != [0]:
                coordinates[c_i][0, arg[0], arg[1]] = 1
                c_i += 1

        action_np = {'function_id': function_id,
                     'coordinate1': coordinates[0],
                     'coordinate2': coordinates[1]}

        assert action_np['function_id'].shape == (573,), "action_np.function_id is in the wrong shape"
        assert action_np['coordinate1'].shape == (1, 64, 64), "action_np.coordinate1 is in the wrong shape"
        assert action_np['coordinate2'].shape == (1, 64, 64), "action_np.coordinate2 is in the wrong shape"

        return action_np

    def sample_batch(self):
        """
        sample a batch of trajectories from the replay buffer and transform them into torch tensors.

        :return: a batch of samples
        """

        transitions = self.replay_buffer.sample()

        """
        The transitions returned from the replay buffer is a dict with the following keys:
        - 'obs_nps': a list of obs_np dict, which dick includes three keys: 'minimap', 'screen' and 'non_spatial'
        - 'action_nps': a list of action_np dict, which dick includes three keys: 
                        'function_id', 'coordinate1' and 'coordinate2'
        - 'returns': a list of length STEPS, each element is a float value
        - 'advantages': a list of length STEPS, each element is a float value
        - 'log_probs': a list of length STEPS, each element is a float value
        """
        # ? make sure the shape of the log_probs

        obs_ts = {'minimap': [], 'screen': [], 'non_spatial': []}
        action_ts = {'function_id': [], 'coordinate1': [], 'coordinate2': []}

        for obs in transitions['obs_nps']:
            for key, value in obs:
                obs_ts[key].append(torch.as_tensor(value, dtype=torch.float32))

        for actions in transitions['action_nps']:
            for key, value in actions:
                action_ts[key].append(torch.as_tensor(value, dtype=torch.float32))

        # return_ts = []
        # advantage_ts = []
        # log_prob_ts = []
        #
        # for i in range(len(transitions['returns'])):
        #     return_ts.append(torch.as_tensor(transitions['returns'][i], dtype=torch.float32))
        #     advantage_ts.append(torch.as_tensor(transitions['advantages'][i], dtype=torch.float32))
        #     log_prob_ts.append(torch.as_tensor(transitions['log_probs'][i], dtype=torch.float32))

        return_ts = torch.as_tensor(transitions['returns'], dtype=torch.float32).to(self.device)
        advantage_ts = torch.as_tensor(transitions['advantages'], dtype=torch.float32).to(self.device)
        log_prob_ts = torch.as_tensor(transitions['log_probs'], dtype=torch.float32).to(self.device)

        # * implement the advantage normalization trick
        advantage_mean = torch.mean(advantage_ts)
        advantage_standard = torch.std(advantage_ts)
        advantage_ts = (advantage_ts - advantage_mean) / advantage_standard

        # send the tensors to the target device
        for key in obs_ts.keys():
            obs_ts[key] = torch.stack(obs_ts[key], dim=0).to(self.device)

        for key in action_ts.keys():
            action_ts[key] = torch.stack(action_ts[key], dim=0).to(self.device)

        transitions = {'obs_ts': obs_ts, 'action_ts': action_ts, 'return_ts': return_ts, 'advantage_ts': advantage_ts,
                       'log_prob_ts': log_prob_ts}

        return transitions

    def compute_actor_loss(self, transitions):
        """
        The function to compute the loss of the actor.

        :param transitions: the transitions returned from the replay buffer
        :return: the loss of actor and the approximal kl
        """
        function_id_distribution, log_probs_new = self.actor(transitions['obs_tss'], transitions['action_tss'])
        ratio = torch.exp(log_probs_new - transitions['log_probs'])
        clipped_advantage = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * transitions['advantages']
        actor_loss = -(torch.min(ratio * transitions['advantages'], clipped_advantage)).mean()

        approx_kl = (transitions['log_probs'] - log_probs_new).mean().item()

        return actor_loss, approx_kl

    def compute_critic_loss(self, transitions):
        """
        The function to compute the loss of the critic.

        :param transitions: the transitions returned from the replay buffer
        :return: the loss of the critic
        """
        return ((self.critic(transitions['obs_tss']) - transitions['returns']) ** 2).mean()

    def optimize(self):
        """
        The function to optimize the module
        """

        transitions = self.sample_batch()

        # back-propagate the loss of the actor
        for i in range(self.actor_train_iterations):
            self.actor_optimizer.zero_grad()
            actor_loss, approx_kl = self.compute_actor_loss(transitions)
            if approx_kl > 1.5 * self.target_kl:
                print('Early stopping at step {} due to reaching max kl.'.format(i))
                break
            actor_loss.backward()
            self.actor_optimizer.step()

        # back-propagate the loss of the critic
        for i in range(self.critic_train_iterations):
            self.critic_optimizer.zero_grad()
            critic_loss = self.compute_critic_loss(transitions)
            critic_loss.backward()
            self.critic_optimizer.step()

    def learn(self, epochs=1000):
        """
        The function to do training.
        :param epochs: number of epochs to train
        """

        best_epoch_reward = 0
        best_epoch_reward_time = 0

        # ! epoch is terminated by the sign that state.last() is True
        for epoch in range(epochs):
            # ! trajectory is terminated by the sign that the agent completed a target (e.g. the agent reached the
            # ! beacon in MoveToBeacon mini-game), or the total steps reached the length limit

            state = self.env.reset()[0]
            obs_np = self._state_2_obs_np(state)

            trajectory_steps = 0

            while True:
                # TODO: to compute the function_call, value and the log_prob
                function_call, value, log_prob = self.actor.step(obs_np)
                state_next = self.env.step(actions=[function_call])[0]
                action_np = self._function_call_2_action_np(function_call)

                self.replay_buffer.store(obs_np, action_np, state_next.reward, value, log_prob)

                trajectory_steps += 1

                obs_next_np = self._state_2_obs_np(state_next)

                obs_np = copy.deepcopy(obs_next_np)
                # ? state = copy.deepcopy(state_next)

                # there are three different terminal conditions:
                # * epoch_terminal represents the whole epoch terminating
                epoch_terminal = state_next.last()
                # * trajectory_terminal represents that the agent reached some target
                # * for now, we only consider that getting a positive reward means one target has been completed
                trajectory_terminal = state_next.reward > 0
                # * trajectory_timeout represents that the rollout/trajectory has reached the maximal steps limit
                trajectory_timeout = trajectory_steps == self.max_trajectory_length

                if epoch_terminal or trajectory_terminal or trajectory_timeout:

                    # * if the trajectory is cut off by the maximal steps limit or the end of the epoch,
                    # * bootstrap the value target
                    if trajectory_timeout or (epoch_terminal and not trajectory_terminal):
                        _, value, _ = self.actor.step(obs_np)
                    else:
                        value = 0

                    self.replay_buffer.complete_trajectory(value)

                    if trajectory_terminal:
                        trajectory_steps = 0

                    if epoch_terminal:
                        # ! only optimize the model after completing one epoch
                        self.optimize()
                        # ! clear the replay buffer after one-step optimization
                        self.replay_buffer.clear()

                        epoch_reward = state.observation['score_cumulative'][0]
                        self.epoch_rewards.append(epoch_reward)

                        # save the best model
                        if epoch_reward > best_epoch_reward:
                            self.save_models(token='best')
                            best_epoch_reward = epoch_reward
                            best_epoch_reward_time = 1
                        if epoch_reward == best_epoch_reward:
                            best_epoch_reward_time += 1

                        print("Episode: \033[34m{}\033[0m, cumulative rewards:\033[32m{}\033[0m, best rewards: "
                              "\033[35m{}\033[0m with \033[33m{}\033[0m times".format(epoch + 1,
                                                                                      epoch_reward,
                                                                                      best_epoch_reward,
                                                                                      best_epoch_reward_time))
                        break

            # save check-points
            if (epoch + 1) % self.check_point_save_epochs == 0:
                self.save_models(token="{}".format(epoch + 1))

        self.env.close()

    def save_models(self, token=''):
        """
        The function to save the target actor and critic networks

        :param token: a token to identify the model
        """

        save_path = os.path.join(self.save_path, token)
        os.makedirs(save_path, exist_ok=True)

        torch.save(self.actor.state_dict(), os.path.join(save_path, 'actor.pt'))
        torch.save(self.critic.state_dict(), os.path.join(save_path, 'critic.pt'))

        np.save(os.path.join(save_path, "epoch_rewards.npy"), self.epoch_rewards)

        print('Model and Information with token-{} saved successfully'.format(token))

    def load_models(self, token=''):
        """
        The function to load the target actor and critic networks, and copy them onto actor and critic networks

        :param token: the token to identify the model
        :return:
        """

        model_path = os.path.join(self.save_path, token)

        self.actor.load_state_dict(torch.load(os.path.join(model_path, self.model_name + "_" + token + '_actor.pt')))
        self.critic.load_state_dict(torch.load(os.path.join(model_path, self.model_name + "_" + token + '_critic.pt')))

        print('Models loaded successfully')

    def restore(self, token, episodes=1000):
        """
        The function to restore the training.

        :param token: the token to identify the model
        :param episodes: number of episodes to continue training
        :return:
        """

        self.load_models(token)
        self.learn(episodes)
