import torch
import gym
import numpy as np
from DDPG import DDPG
from utils import ReplayBuffer
from utils import Tracker

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
    
    ############### Hyperparameters ################
    env_name = "MountainCarContinuous-v0"
    max_episodes = 5             # max num of episodes to render
    random_seed = 0
    render = False
    delay = 2000000             # loop delay b/w frames for rendering
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # primitive action parameters
    action_bounds = env.action_space.high[0]
    action_offset = np.array([0.0])
    action_offset = torch.FloatTensor(action_offset.reshape(1, -1)).to(device)
    
    # state parameters
    state_bounds_np = np.array([0.9, 0.07])
    state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    state_offset =  np.array([-0.3, 0.0])
    state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)

    goal_state = np.array([0.48, 0.04])        # final goal state to be achived
    threshold = np.array([0.01, 0.02])         # threshold value to check if goal state is achieved
    
    # HAC parameters:
    k_level = 2                     # num of levels in hierarchy
    H = 20                          # time horizon to achieve subgoal
    
    # DDPG parameters:
    lr = 0.001
    
    # save trained models
    directory = "./preTrained/{}/{}level".format(env_name, k_level) 
    filename = "HAC_{}".format(env_name)
    filename = filename + '_solved'
    ####################################################
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # adding lowest level
    HAC = [DDPG(state_dim, action_dim, action_bounds, action_offset, lr, H)]
    replay_buffer = [ReplayBuffer()]
    
    # adding remaining levels
    for _ in range(k_level-1):
        HAC.append(DDPG(state_dim, state_dim, state_bounds, state_offset, lr, H))
        replay_buffer.append(ReplayBuffer())
    
    # load policy:
    for i in range(k_level):
        HAC[i].load_actor(directory, filename+'_level_{}'.format(i))
        
    # logging variable
    tracker = Tracker(k_level)
    
    
    #   <================ HAC functions ================>
    
    def check_goal(state, goal, threshold):
        for i in range(state_dim):
            if abs(state[i]-goal[i]) > threshold[i]:
                return False
        return True
    
    def run_HAC_test(i_level, state, goal):
        next_state = None
        done = None
        
        tracker.goals[i_level] = goal
        
        for attempt in range(H):
            action = HAC[i_level].select_action(state, goal)
            
            #   <================ high level policy ================>
            if i_level > 0:
                next_state, done = run_HAC_test(i_level-1, state, action)
                
                
            #   <================ low level policy ================>
            else:
                # take primitive action
                next_state, re, done, _ = env.step(action)
                
                if render:
                    # env.render()
                    
                    if k_level == 2:
                        env.render_goal(tracker.goals[0], tracker.goals[1])
                    elif k_level == 3:
                        env.render_goal_2(tracker.goals[0], tracker.goals[1], tracker.goals[2])
                    
                # logging
                tracker.reward += re
                tracker.timestep +=1
                
                # delay for slowing render
                for _ in range(delay):
                    continue
            
            state = next_state
            
            # check if goal is achieved
            goal_achieved = check_goal(next_state, goal, threshold)
            
            if done or goal_achieved:
                break
            
        return next_state, done
        
    
    #   <================ evaluation ================>
    
    for i_episode in range(1, max_episodes+1):
        
        tracker.reward = 0
        tracker.timestep = 0
        
        state = env.reset()
        run_HAC_test(k_level-1, state, goal_state)
        
        print("Episode: {}\t Reward: {}\t len: {}".format(i_episode, tracker.reward, tracker.timestep))
    
    env.close()


if __name__ == '__main__':
    test()
 
  
