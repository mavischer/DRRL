# A2C training of Relational Deep Reinforcement Learning Architecture

## Introduction

Torch implementation of the deep relational architecture from the paper ["Relational Deep Reinforcement Learning"](https://arxiv.org/pdf/1806.01830.pdf) together with (synchronous) advantage-actor-critic training as discussed for example [here](https://arxiv.org/abs/1602.01783).

The Box-World environment used in this script can be found at [this repo](https://github.com/mavischer/Box-World).

## Example Run

```bash
python a2c.py -c exmpl_config.yml -s example_run
```