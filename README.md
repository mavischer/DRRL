# A2C training of Relational Deep Reinforcement Learning Architecture

## Introduction

Torch implementation of the deep relational architecture from the paper ["Relational Deep Reinforcement Learning"](https://arxiv.org/pdf/1806.01830.pdf) together with (synchronous) advantage-actor-critic training as discussed for example [here](https://arxiv.org/abs/1602.01783).

The Box-World environment used in this script can be found at [this repo](https://github.com/mavischer/Box-World).

Training is performed in `a2c.py`, the relational module and general architecture are both implemented as `torch.nn.Module` in `attention_module.py`.
An example YAML config file parsed from the arguments is `exmpl_config.yml`. Both environment and network can be parameterized there.
A suitable environment can be created using `requirements.txt` or `environment.yml`. Afterwards install and register the [Box-World environment](https://github.com/mavischer/Box-World) by cloning the repo and `pip install -e gym-boxworld`.

The network's graph can be visualized using tensorboard.

## Example Run

```bash
python a2c.py -c exmpl_config.yml -s example_run
```