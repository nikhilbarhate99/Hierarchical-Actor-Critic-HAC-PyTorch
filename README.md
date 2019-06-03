# Hierarchical-Actor-Critic-HAC-PyTorch

This is an implementation of the Hierarchical Actor Critic (HAC) algorithm described in the paper, [Learning Multi-Level Hierarchies with Hindsight](https://arxiv.org/abs/1712.00948), in PyTorch for OpenAI gym environments.

- All the hyperparameters are contained in the `HAC.py` file.
- To train a new network run `HAC.py`
- To test a preTrained network run `test.py`

To render the environment with subgoals (2 or 3 level) replace the gym files in local installation directory `gym/envs/classic_control` with the files in gym folder of this repo.


## Requirements

- Python 3.6
- [PyTorch](https://pytorch.org/)
- [OpenAI gym](https://gym.openai.com/)

## Disclaimer

The results are not consistent with the current hyperparameters. I will update the repo if I find stable hyperparameters.


## Results

MountainCarContinuous-v0 (2 levels, 300 episodes)  |
:-----------------------------------:|
![](https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/gif/MountainCarContinuous-v0.gif)  |





## References

- Original [Paper](https://arxiv.org/abs/1712.00948) and [Code](https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-)
