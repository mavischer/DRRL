# A2C training of Relational Deep Reinforcement Learning Architecture

## Introduction

Torch implementation of the deep relational architecture from the paper ["Relational Deep Reinforcement Learning"](https://arxiv.org/pdf/1806.01830.pdf) together with (synchronous) advantage-actor-critic training as discussed for example [here](https://arxiv.org/abs/1602.01783).

The Box-World environment used in this script can be found at [this repo](https://github.com/mavischer/Box-World).

Training is performed in `a2c_fast.py`. The implementation is based on [this repo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) which turned out to be more clever and substantially faster than my own implementation `a2c_dist.py`.
However this latter file contains routines to plot the gradients in the network and the computation graph.

The relational module and general architecture are both implemented as `torch.nn.Module` in `attention_module.py`. However, `a2c_fast.py` uses almost identical adaptations of these classes in `helper/a2c_ppo_acktr/model.yml` that comply with the training algorithm's `Policy` class.

An example YAML config file parsed from the arguments is `configs/exmpl_config.yml`. Training, the environment and network can be parameterized there. A copy of the loaded configuration file will be saved with checkpoints and logs for documentation.

A suitable environment can be created e.g. by  `conda env create -f environment.yml` or 
 `pip install -r requirements.txt`. Afterwards install and register the [Box-World environment](https://github.com/mavischer/Box-World) by cloning the repo and `pip install -e gym-boxworld`.
*Remember that after changing the code you need to re-register the environment before the changes become effective.*
You can find the details of state space, action space and reward structure there.

`visualize_results.ipynb` contains some plotting functionality.

## Example Run

```bash
python a2c.py -c configs/exmpl_config.yml -s example_run
```