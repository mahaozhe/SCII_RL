"""
Deep Deterministic Policy Gradient (DDPG) Algorithm.

Reference:
- [DDPG] Continuous control with deep reinforcement learning: https://arxiv.org/abs/1509.02971
- OpenAI Spinning Up implemented DDPG algorithm: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
"""

import numpy as np
from pysc2.lib import actions as ACTIONS
import torch
import torch.nn as nn
from torch.nn.functional import gumbel_softmax
from torch.distributions import Categorical

import os
import copy

from utils.ReplayBuffer import DDPGReplayBuffer


class DDPG:
    """
    The Deep Deterministic Policy Gradient (DDPG) Agent.
    """

    # TODO: abstract a base class

    def __init__(self, env, actor, critic, replay_buffer_size=10000, device=None, actor_lr=0.001, critic_lr=0.001,
                 gamma=0.99, tau=0.005, batch_size=32, warmup_steps=1000, soft_update_steps=1000, map_size=64, seed=0,
                 action_space=len(ACTIONS.FUNCTIONS), save_path="./Saves/", model_name='UnnamedModel', save_epochs=100):
        """
        Initialization.

        :param env: the SC2Env environment instance
        :param actor: the actor network
        :param critic: the critic network
        :param replay_buffer_size: the capacity of the replay buffer
        :param device: the training device
        :param actor_lr: the learning rate of the actor model
        :param critic_lr: the learning rate of the critic model
        :param gamma: the discount factor
        :param tau: the soft update factor
        :param batch_size: batch size
        :param warmup_steps: warm up steps
        :param soft_update_steps: how many steps to update the target networks
        :param map_size: the size of the map
        :param seed: a random seed
        :param action_space: the length of the action space
        :param save_path: the location to save the model
        :param model_name:a name for the training model
        :param save_epochs: how many epochs to save one check point
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
        self.target_actor = copy.deepcopy(actor).to(self.device)

        # initialize the policy critic and the target critic
        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(critic).to(self.device)

        # the target networks are used to evaluate
        self.target_actor.eval()
        self.target_critic.eval()

        # initialize two optimizers using Adam
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)

        # to record how many iterations
        self.iteration = 0

        # initialize the replay buffer
        self.replay_buffer = DDPGReplayBuffer(replay_buffer_size)

        # specify the action space, a number to identify the number of possible actions
        self.action_space = action_space
        # for the map size
        self.map_size = map_size

        # for the batch size of training
        self.batch_size = batch_size

        # for the discount factor
        self.gamma = gamma

        # for the soft update
        self.tau = tau
        self.soft_update_steps = soft_update_steps

        # for training warm-up steps
        self.warmup_steps = warmup_steps

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

    def _action_ts_2_function_call(self, action_ts, available_actions):
        """
        The helper to transform an action dict with tensor values to FunctionCall to interact with SC2Env.

        :param action_ts: an action dict with tensor values
        :param available_actions: the available actions for this step
        :return: an SC2Env-interacted FunctionCall instance
        """

        probable_function_id = nn.Softmax(dim=-1)(action_ts['function_id']).detach()
        probable_function_id = probable_function_id * available_actions

        # * make sure the sum of all probabilities is 1
        if probable_function_id.sum(1) == 0:
            # ? select an available_action uniformly
            # ? BUT, why the probabilities are all 0?
            distribution = Categorical(available_actions)
        else:
            distribution = Categorical(probable_function_id)

        # sample the function id from distribution
        function_id = distribution.sample().item()

        # sample the coordinates
        coordinate_position1 = nn.Softmax(dim=-1)(action_ts['coordinate1'].view(1, -1)).detach()
        coordinate_position1 = Categorical(coordinate_position1).sample().item()

        coordinate_position2 = nn.Softmax(dim=-1)(action_ts['coordinate2'].view(1, -1)).detach()
        coordinate_position2 = Categorical(coordinate_position2).sample().item()

        positions = [[int(coordinate_position1 % self.map_size), int(coordinate_position1 // self.map_size)],
                     [int(coordinate_position2 % self.map_size), int(coordinate_position2 // self.map_size)]]

        # put the position coordinates into the args list
        args = []
        number_of_arg = 0
        for arg in ACTIONS.FUNCTIONS[function_id].args:
            if arg.name in ['screen', 'screen2', 'minimap']:
                args.append(positions[number_of_arg])
                number_of_arg += 1
            else:
                # for now, for other kinds of arguments (such as `queued`), give [0] by default
                args.append([0])

        return ACTIONS.FunctionCall(function_id, args)

    def _obs_np_2_obs_ts(self, obs_np):
        """
        The helper to transform an obs dict with numpy values to an obs dict with tensor values and move to the device.

        :param obs_np: an obs dict with numpy value
        :return: an obs dict with tensor value
        """

        obs_ts = {}

        for key in obs_np.keys():
            x = obs_np[key].astype('float32')
            x = np.expand_dims(x, 0)
            obs_ts[key] = torch.from_numpy(x).to(self.device)

        return obs_ts

    def _gumbel_softmax(self, x):
        """
        A helper to do gumbel softmax

        :param x: input vector
        :return: output normalized vector
        """

        shape = x.shape

        if len(shape) == 4:
            x_reshape = x.contiguous().view(shape[0], -1)
            y = gumbel_softmax(x_reshape, hard=True, dim=-1)
            y = y.contiguous().view(shape)
        else:
            y = gumbel_softmax(x, hard=True, dim=-1)

        return y

    def sample_batch(self):
        """
        sample a batch of trajectories from the replay buffer and transform them into torch tensors.

        :return: a batch of samples
        """

        transitions = self.replay_buffer.sample(self.batch_size)

        obs_ts = {'minimap': [], 'screen': [], 'non_spatial': []}
        action_ts = {'function_id': [], 'coordinate1': [], 'coordinate2': []}
        reward_ts = []
        obs_next_ts = {'minimap': [], 'screen': [], 'non_spatial': []}
        done_ts = []

        for transition in transitions:
            for key, value in transition.obs.items():
                value = torch.as_tensor(value, dtype=torch.float32)
                obs_ts[key].append(value)

            for key, value in transition.action.items():
                value = torch.as_tensor(value, dtype=torch.float32)
                action_ts[key].append(value)

            for key, value in transition.obs_next.items():
                value = torch.as_tensor(value, dtype=torch.float32)
                obs_next_ts[key].append(value)

            reward_ts.append(torch.as_tensor(transition.reward, dtype=torch.float32))

            done_ts.append(torch.as_tensor(1 if transition.done else 0, dtype=torch.float32))

        for key in obs_ts.keys():
            obs_ts[key] = torch.stack(obs_ts[key], dim=0).to(self.device)
            obs_next_ts[key] = torch.stack(obs_next_ts[key], dim=0).to(self.device)

        for key in action_ts.keys():
            action_ts[key] = torch.stack(action_ts[key], dim=0).to(self.device)

        reward_ts = torch.tensor(reward_ts).to(self.device)
        done_ts = torch.tensor(done_ts).to(self.device)

        return obs_ts, action_ts, obs_next_ts, reward_ts, done_ts

    def select_action_from_obs_np(self, obs_np):
        """
        The function to return a FunctionCall with arguments given a state from SC2Env

        :param state: a state from SC2Env
        :return: a pysc2 FunctionCall as an action
        """

        obs_ts = self._obs_np_2_obs_ts(obs_np)

        # return logit-actions from the actor
        action_ts = self.actor(obs_ts)

        available_actions_now = obs_ts['non_spatial']

        function_call = self._action_ts_2_function_call(action_ts, available_actions_now)

        return function_call

    def soft_update(self, target, source, tau):
        """
        The function to implement the soft update, for hard update, set tau = 1

        :param target: target network
        :param source: original training network
        :param tau: soft update factor, usually near to 0
        """

        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + source_param.data * tau)

    def optimize(self):
        """
        Sample a batch of data from the replay buffer and don the optimization
        """

        obs_0_ts, action_0_ts, obs_1_ts, reward, done = self.sample_batch()

        # optimize the critic
        # using the target actor network, here the forward function only outputs the logits
        actor_action_by_obs_1 = self.target_actor(obs_1_ts)

        action_1_ts = {}
        for key, value in actor_action_by_obs_1.items():
            action_1_ts[key] = self._gumbel_softmax(value)

        q_next = self.target_critic(obs_1_ts, action_1_ts).detach()
        q_next = torch.squeeze(q_next)

        # compute the TD error:
        # y_expected = reward + \gamma * Q'(state_1, \pi'(state_1))
        # y_predicted = Q(state_0, action_0)
        y_expected = reward + self.gamma * q_next * (1 - done)
        y_predicted = self.critic(obs_0_ts, action_0_ts)
        y_predicted = torch.squeeze(y_predicted)

        # compute the critic loss
        loss_critic = nn.SmoothL1Loss()(y_predicted, y_expected)

        # update critic
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # optimize the actor
        actor_action_by_obs_0 = self.actor(obs_0_ts)

        action_0_predicted_ts = {}
        for key, value in actor_action_by_obs_0.items():
            action_0_predicted_ts[key] = self._gumbel_softmax(value)

        # compute the loss
        l2_reg = torch.FloatTensor(1).to(self.device)
        for param in self.actor.parameters():
            l2_reg = l2_reg + param.norm(2)

        q_max = self.critic(obs_0_ts, action_0_predicted_ts)
        q_max = -1 * q_max.mean()

        loss_actor = q_max + torch.squeeze(l2_reg) * 1e-3

        # update the actor
        self.actor.train()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # update the target network
        if self.iteration % self.soft_update_steps == 0:
            self.soft_update(self.target_actor, self.actor, self.tau)
            self.soft_update(self.target_critic, self.critic, self.tau)

    def learn(self, epochs=1000):
        """
        The function to do training.
        :param epochs: number of epochs to train
        """

        best_epoch_reward = 0
        best_epoch_reward_time = 0

        for epoch in range(epochs):
            state = self.env.reset()[0]
            obs_np = self._state_2_obs_np(state)

            while True:
                function_call = self.select_action_from_obs_np(obs_np)

                state_next = self.env.step(actions=[function_call])[0]

                action_np = self._function_call_2_action_np(function_call)

                obs_next_np = self._state_2_obs_np(state_next)

                self.replay_buffer.store(obs_np, action_np, obs_next_np, state.reward, state_next.last())

                self.iteration += 1

                if self.iteration > self.warmup_steps:
                    # optimize the model after the warmup steps
                    self.optimize()

                if state_next.last():
                    epoch_reward = state_next.observation['score_cumulative'][0]
                    self.epoch_rewards.append(epoch_reward)

                    # save the best model
                    if epoch_reward > best_epoch_reward:
                        self.save_models(token='best')
                        best_epoch_reward = epoch_reward
                        best_epoch_reward_time = 1
                    elif epoch_reward == best_epoch_reward:
                        best_epoch_reward_time += 1

                    print("Epoch: \033[34m{}\033[0m, epoch rewards:\033[32m{}\033[0m, best rewards: \033[35m{}\033[0m "
                          "with \033[33m{}\033[0m times".format(epoch + 1,
                                                                epoch_reward,
                                                                best_epoch_reward,
                                                                best_epoch_reward_time))
                    break

                else:
                    obs_np = copy.deepcopy(obs_next_np)
                    state = copy.deepcopy(state_next)

            # save a check-point
            if (epoch + 1) % self.check_point_save_epochs == 0:
                self.save_models(token="{}".format(epoch + 1))

        # before the training completed, update the target networks once again.
        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)

        self.save_models(token='final')
        self.env.close()

        print("Training Completed!")

    def save_models(self, token=''):
        """
        The function to save the target actor and critic networks

        :param token: a token to identify the model
        """

        save_path = os.path.join(self.save_path, token)
        os.makedirs(save_path, exist_ok=True)

        torch.save(self.target_actor.state_dict(), os.path.join(save_path, 'actor.pt'))
        torch.save(self.target_critic.state_dict(), os.path.join(save_path, 'critic.pt'))

        np.save(os.path.join(save_path, "epoch_rewards.npy"), self.epoch_rewards)

        print('Model and Information with token-{} saved successfully'.format(token))

    def load_models(self, token=''):
        """
        The function to load the target actor and critic networks, and copy them onto actor and critic networks

        :param token: the token to identify the model
        """

        model_path = os.path.join(self.save_path, token)

        self.actor.load_state_dict(torch.load(os.path.join(model_path, 'actor.pt')))
        self.critic.load_state_dict(torch.load(os.path.join(model_path, 'critic.pt')))

        self.soft_update(self.target_actor, self.actor, 1)
        self.soft_update(self.target_critic, self.critic, 1)

        print('Models loaded successfully')

    def restore(self, token, episodes=1000, restore_token=1):
        """
        The function to restore the training.

        :param token: the token to identify the model
        :param episodes: number of episodes to continue training
        :param restore_token: a token to identify the number of restores, if it (N) larger than 1,
                                then load the model from "self.save_path/restore-(N-1)/token/"
        """

        assert isinstance(restore_token, int), "the restore_token parameter is NOT an int value"

        if restore_token == 1:
            self.load_models(token)
        else:
            self.load_models(os.path.join("restore-{}".format(restore_token - 1), token))

        # change the save_path to a new folder
        self.save_path = os.path.join(self.save_path, "restore-{}".format(restore_token))
        self.learn(episodes)
