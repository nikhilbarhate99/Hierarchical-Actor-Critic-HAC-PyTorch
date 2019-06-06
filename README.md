# Hierarchical-Actor-Critic-HAC-PyTorch

This is an implementation of the Hierarchical Actor Critic (HAC) algorithm described in the paper, [Learning Multi-Level Hierarchies with Hindsight](https://arxiv.org/abs/1712.00948), in PyTorch for OpenAI gym environments.

- All the hyperparameters are contained in the `train.py` file.
- To train a new network run `train.py`
- To test a preTrained network run `test.py`

To render the environment with subgoals (2 or 3 level) replace the gym files in local installation directory `gym/envs/classic_control` with the files in gym folder of this repo and change the bool `render` to True


## Requirements

- Python 3.6
- [PyTorch](https://pytorch.org/)
- [OpenAI gym](https://gym.openai.com/)


## Results

### MountainCarContinuous-v0
 (2 levels, H = 20, 200 episodes)  |  (3 levels, H = 5, 200 episodes)  |
:-----------------------------------:|:-----------------------------------:|
![](https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/gif/MountainCarContinuous-v0.gif)  | ![](https://github.com/nikhilbarhate99/Hierarchical-Actor-Critic-HAC-PyTorch/blob/master/gif/MountainCarContinuous-v0-3level.gif)  |





## References

- Original [Paper](https://arxiv.org/abs/1712.00948) and [Code](https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-)
