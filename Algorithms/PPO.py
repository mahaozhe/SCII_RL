"""
Proximal Policy Optimization (by clipping) with early stopping based on approximate KL.

Reference:
- [PPO] Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347
- [GAE] High-Dimensional Continuous Control Using Generalized Advantage Estimation: https://arxiv.org/abs/1506.02438
- OpenAI Spinning Up implemented PPO algorithm: https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""

# TODO: change the log_prob

import numpy as np
from pysc2.lib import actions as ACTIONS
import torch
import torch.nn as nn
from torch.distributions import Categorical

import os
import copy

from utils.ReplayBuffer import PPOReplayBuffer


class PPO:
    """
    The Proximal Policy Optimization (by clipping) agent
    """

    # TODO: abstract a base class

    def __init__(self, env, actor, critic, device=None, gamma=0.99, clip_ratio=0.2, actor_lr=0.0003, critic_lr=0.001,
                 actor_train_iterations=80, critic_train_iterations=80, lamb=0.97, max_trajectory_length=1000,
                 target_kl=0.01, seed=0, action_space=len(ACTIONS.FUNCTIONS), map_size=64, warmup_steps=1000,
                 save_path="./Saves/", model_name='UnnamedModel', save_epochs=100):
        """
        Initialization.

        :param env: the SC2Env environment instance
        :param actor: the actor network
        :param critic: the critic network
        :param device: the training device
        :param gamma: the discount factor
        :param clip_ratio: the clip ratio
        :param actor_lr: the learning rate of the actor model
        :param critic_lr: the learning rate of the critic model
        :param actor_train_iterations: the number of actor training iterations
        :param critic_train_iterations: the number of critic training iterations
        :param lamb: the lambda for the GAE
        :param max_trajectory_length: the maximum length of one trajectory
        :param target_kl: the target kl limit
        :param seed: the random seed
        :param action_space: the length of the action space
        :param map_size: the size of the map
        :param warmup_steps: the warm up steps?
        :param save_path: the location to save the model
        :param model_name: a name for the training model
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

        # specify the action space, a number to identify the number of possible actions
        self.action_space = action_space
        # for the map size
        self.map_size = map_size

        # ? for training warm-up steps
        # self.warmup_steps = warmup_steps
        # to record how many iterations
        # self.iteration = 0

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

        function_id = distribution.sample()

        # ! to compute the log_prob here
        log_prob = {'function_id': distribution.log_prob(function_id).item()}
        function_id = function_id.item()

        coordinate_position1 = nn.Softmax(dim=-1)(action_ts['coordinate1'].view(1, -1)).detach()
        coordinate_position1_distribution = Categorical(coordinate_position1)
        coordinate_position1 = coordinate_position1_distribution.sample()
        log_prob['coordinate1'] = coordinate_position1_distribution.log_prob(coordinate_position1).item()
        coordinate_position1 = coordinate_position1.item()

        coordinate_position2 = nn.Softmax(dim=-1)(action_ts['coordinate2'].view(1, -1)).detach()
        coordinate_position2_distribution = Categorical(coordinate_position2)
        coordinate_position2 = coordinate_position2_distribution.sample()
        log_prob['coordinate2'] = coordinate_position2_distribution.log_prob(coordinate_position2).item()
        coordinate_position2 = coordinate_position2.item()

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

        return ACTIONS.FunctionCall(function_id, args), log_prob

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

    def _select_function_call(self, obs_np):
        """
        The function to select the function call based on the current actor model

        :param obs_np: the observation in numpy array style
        :return: a pysc2 FunctionCall with arguments and the log probability of the selected action
        """

        # * transform the obs_np to obs_ts before feeding it to the actor model
        obs_ts = self._obs_np_2_obs_ts(obs_np)

        # ! no need the with statement, because we are not evaluating
        action_ts, _ = self.actor(obs_ts)

        available_actions_now = obs_ts['non_spatial']

        function_call, log_prob = self._action_ts_2_function_call(action_ts, available_actions_now)

        return function_call, log_prob

    def _compute_value_given_obs_np(self, obs_np):
        """
        The function to compute the value of the current state given the obs_np based on the current critic model

        :param obs_np: the observation in numpy style\
        :return: the estimated value of the current state
        """

        # * transform the obs_np to obs_ts before feeding it to the actor model
        obs_ts = self._obs_np_2_obs_ts(obs_np)

        # ! no need the with statement, because we are not evaluating
        value = self.critic(obs_ts)

        return value

    def sample_batch(self):
        """
        sample a batch of trajectories from the replay buffer and transform them into torch tensors.

        :return: a batch of samples
        """

        transitions = self.replay_buffer.sample()

        """
        The transitions returned from the replay buffer is a dict with the following keys:
        - 'obs_nps': a list of obs_np dict, which includes three keys: 'minimap', 'screen' and 'non_spatial'
        - 'action_nps': a list of action_np dict, each includes three keys: 
                        'function_id', 'coordinate1' and 'coordinate2'
        - 'returns': a list of length STEPS, each element is a float value
        - 'advantages': a list of length STEPS, each element is a float value
        - 'log_probs': a list of log_prob dict, each includes three keys: 'function_id', 'coordinate1' and 'coordinate2'
        """

        obs_ts = {'minimap': [], 'screen': [], 'non_spatial': []}
        action_ts = {'function_id': [], 'coordinate1': [], 'coordinate2': []}
        log_prob_ts = {'function_id': [], 'coordinate1': [], 'coordinate2': []}

        for obs in transitions['obs_nps']:
            for key, value in obs.items():
                obs_ts[key].append(torch.as_tensor(value, dtype=torch.float32))

        for actions in transitions['action_nps']:
            for key, value in actions.items():
                action_ts[key].append(torch.as_tensor(value, dtype=torch.float32))

        for log_probs in transitions['log_probs']:
            for key, value in log_probs.items():
                log_prob_ts[key].append(torch.as_tensor(value, dtype=torch.float32))

        return_ts = torch.as_tensor(transitions['returns'], dtype=torch.float32).to(self.device)
        advantage_ts = torch.as_tensor(transitions['advantages'], dtype=torch.float32).to(self.device)

        # * implement the advantage normalization trick
        advantage_mean = torch.mean(advantage_ts)
        advantage_standard = torch.std(advantage_ts)
        advantage_ts = (advantage_ts - advantage_mean) / advantage_standard

        # send the tensors to the target device
        for key in obs_ts.keys():
            obs_ts[key] = torch.stack(obs_ts[key], dim=0).to(self.device)

        for key in action_ts.keys():
            action_ts[key] = torch.stack(action_ts[key], dim=0).to(self.device)
            # * the log_prob_ts has the same keys as action_ts
            log_prob_ts[key] = torch.stack(log_prob_ts[key], dim=0).to(self.device)

        transitions = {'obs_ts': obs_ts, 'action_ts': action_ts, 'return_ts': return_ts, 'advantage_ts': advantage_ts,
                       'log_prob_ts': log_prob_ts}

        return transitions

    def compute_actor_loss(self, transitions):
        """
        The function to compute the loss of the actor.

        :param transitions: the transitions returned from the replay buffer
        :return: the loss of actor and the approximated kl
        """

        _, log_probs_new = self.actor(transitions['obs_ts'], transitions['action_ts'])

        # * the ratio and the approx_kl is the average over three keys: 'function_id', 'coordinate1' and 'coordinate2'
        ratio = torch.zeros(log_probs_new['function_id'].size()[0], requires_grad=True).to(self.device)
        for key in log_probs_new:
            ratio += torch.exp(log_probs_new[key] - transitions['log_prob_ts'][key])
        ratio /= 3

        # ratio = torch.exp(log_probs_new - transitions['log_prob_ts'])
        clipped_advantage = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * transitions['advantage_ts']
        actor_loss = -(torch.min(ratio * transitions['advantage_ts'], clipped_advantage)).mean()

        approx_kl = torch.zeros(log_probs_new['function_id'].size()[0], requires_grad=True).to(self.device)
        for key in log_probs_new:
            approx_kl += transitions['log_prob_ts'][key] - log_probs_new[key]
        approx_kl = (approx_kl / 3).mean().item()

        return actor_loss, approx_kl

    def compute_critic_loss(self, transitions):
        """
        The function to compute the loss of the critic.

        :param transitions: the transitions returned from the replay buffer
        :return: the loss of the critic
        """
        return ((self.critic(transitions['obs_ts']) - transitions['return_ts']) ** 2).mean()

    def optimize(self):
        """
        The function to optimize the module
        """

        transitions = self.sample_batch()

        # back-propagate the loss of the actor
        for i in range(self.actor_train_iterations):
            self.actor_optimizer.zero_grad()
            actor_loss, approx_kl = self.compute_actor_loss(transitions)
            if approx_kl > self.target_kl:
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

                function_call, log_prob = self._select_function_call(obs_np)
                value = self._compute_value_given_obs_np(obs_np)

                state_next = self.env.step(actions=[function_call])[0]
                action_np = self._function_call_2_action_np(function_call)

                self.replay_buffer.store(obs_np, action_np, state_next.reward, value.item(), log_prob)

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
                        value = self._compute_value_given_obs_np(obs_np).item()
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

                        epoch_reward = state_next.observation['score_cumulative'][0]
                        self.epoch_rewards.append(epoch_reward)

                        # save the best model
                        if epoch_reward > best_epoch_reward:
                            self.save_models(token='best')
                            best_epoch_reward = epoch_reward
                            best_epoch_reward_time = 1
                        elif epoch_reward == best_epoch_reward:
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

        self.save_models(token='final')
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

        self.actor.load_state_dict(torch.load(os.path.join(model_path, 'actor.pt')))
        self.critic.load_state_dict(torch.load(os.path.join(model_path, 'critic.pt')))

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
