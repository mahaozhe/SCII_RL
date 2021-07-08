# Some Flat RL Algorithms on StartCraft II Mini-Games

The project implemented some flat reinforcement learning algorithms on *StarCraft II* mini-games based
on [PySC2 Lib](https://github.com/deepmind/pysc2)

## Installation

* install PyTorch based on your own configuration: [PyTorch Installation Link](https://pytorch.org/get-started/)
* install all packages in `requirements.txt`: `pip install -r requirements.txt`

## Project Structure

- [main.py](./main.py): instantiate the Agent/Algorithm and an `SC2Env`, then run the training or evaluation
- [Algorithms](./Algorithms/): algorithms/agents to learn and test
- [Networks](./Networks/): neural networks
- [utils](./utils/): some helpers
- [NotRelated](./NotRelated/): some other files that are not related to the project

## Notes (TEMP):

In the project:

- An `obs` is a dict with three keys: `minimap`, `screen` and `non_spatial`.
    - There are two kinds of `obs` with *numpy array* and *PyTorch tensor* data type, which are represented with `_np`
      and `_ts` suffix respectively.
    - `obs` is only for the agent/algorithm itself, for the states returned from `SC2Env`, we are using `state`.
- An `action` is dict with three keys: `function_id`, `coordinate1` and `coordinate2`.
    - There are two kinds of `action` with *numpy array* and *PyTorch tensor* data type, which are represented
      with `_np` and `_ts` suffix respectively.
    - **For now, for all Actor-Critic algorithms**, the critic network only predicts the logits without a softmax layer,
      which is put in the `optimize()` function, the `logits` has the same structure as `action`, and **for now**
      , `action` and `logits` are exactly same in the project, and we reserve `action` only.
    - `action` (same as `logits`) is only for the agent/algorithm itself, for the actions to interact with `SC2Env`, we
      are using `function_call`.
- The `non_spatial` argument in `obs` packaged from the state only contains `available_actions` for now.
- The predicted arguments only contains `screen`, `screen2` and `minimap` for now. Based on the experience, no function
  has more than two arguments from them.
- We assume the height and the width of the minimap and the screen are the same.
- The models or checkpoints save in `save_path/model_name/token/MODELS_AND_INFORMATION` using the token to identify a
  single model or checkpoint
- For some on-line algorithms that using trajectories instead of transitions, we define that finishing one target as a
  trajectory, and finishing the episode as an epoch.
- An `log_prob(s)` is a dict with three keys: `function_id`, `coordinate1` and `coordinate2`.