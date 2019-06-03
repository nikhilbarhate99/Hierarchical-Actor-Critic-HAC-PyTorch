import torch
import gym
import numpy as np
from DDPG import DDPG
from utils import ReplayBuffer
from utils import Tracker

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    #################### Hyperparameters ####################
    env_name = "MountainCarContinuous-v0"
    save_episode = 10               # keep saving every n episodes
    max_episodes = 1000             # max num of training episodes
    random_seed = 0
    render = False
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    """
     Actions (both primitive and subgoal) are implemented as follows:
       action = ( network output (Tanh) * bounds ) + offset
       clip_high and clip_low bound the exploration noise
    """
    
    # primitive action bounds and offset
    action_bounds = env.action_space.high[0]
    action_offset = np.array([0.0])
    action_offset = torch.FloatTensor(action_offset.reshape(1, -1)).to(device)
    action_clip_low = np.array([-1.0 * action_bounds])
    action_clip_high = np.array([action_bounds])
    
    # state bounds and offset
    state_bounds_np = np.array([0.9, 0.07])
    state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    state_offset =  np.array([-0.3, 0.0])
    state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)
    state_clip_low = np.array([-1.2, -0.07])
    state_clip_high = np.array([0.6, 0.07])
    
    # exploration noise std for primitive action and subgoals
    exploration_action_noise = np.array([0.1])        
    exploration_state_noise = np.array([0.02, 0.01]) 
    
    goal_state = np.array([0.48, 0.04])        # final goal state to be achived
    threshold = np.array([0.01, 0.02])         # threshold value to check if goal state is achieved
    
    # HAC parameters:
    k_level = 2                 # num of levels in hierarchy
    H = 20                      # time horizon to achieve subgoal
    lamda = 0.4                 # subgoal testing parameter
    
    # DDPG parameters:
    gamma = 0.95                # discount factor for future rewards
    n_iter = 100                # update policy n_iter times in one DDPG update
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    
    # save trained models
    directory = "./preTrained/{}/{}level/".format(env_name, k_level) 
    filename = "HAC_{}".format(env_name)
    #########################################################
    
    
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
    
    # logging variables:
    tracker = Tracker(k_level)
    log_f = open("log.txt","w+")
    
    #   <================ HAC functions ================>
    
    def check_goal(state, goal, threshold):
        for i in range(state_dim):
            if abs(state[i]-goal[i]) > threshold[i]:
                return False
        return True
    
    def run_HAC(i_level, state, goal, subgoal_test):
        next_state = None
        done = None
        goal_transitions = []
        
        tracker.goals[i_level] = goal
        
        # H attempts
        for _ in range(H):
            next_subgoal_test = subgoal_test
            
            action = HAC[i_level].select_action(state, goal)
            
            #   <================ high level policy ================>
            if i_level > 0:
                # add noise if not subgoal testing
                if not subgoal_test:
                    action = action + np.random.normal(0, exploration_state_noise)
                    action = action.clip(state_clip_low, state_clip_high)
                
                # Determine whether to test subgoal (action)
                if np.random.random_sample() < lamda:
                    next_subgoal_test = True
                
                # Pass subgoal to lower level 
                next_state, done = run_HAC(i_level-1, state, action, next_subgoal_test)
                
                # if subgoal was tested but not achieved, add subgoal testing transition
                if next_subgoal_test and not check_goal(action, next_state, threshold):
                    replay_buffer[i_level].add((state, action, -H, next_state, goal, 0.0, float(done)))
                
                # for hindsight action transition
                action = next_state
                
            #   <================ low level policy ================>
            else:
                # add noise if not subgoal testing
                if not subgoal_test:
                    action = action + np.random.normal(0, exploration_action_noise)
                    action = action.clip(action_clip_low, action_clip_high)
                
                # take primitive action
                next_state, rew, done, _ = env.step(action)
                
                if render:
                    # env.render()
                    
                    if k_level == 2:
                        env.render_goal(tracker.goals[0], tracker.goals[1])
                    elif k_level == 3:
                        env.render_goal_2(tracker.goals[0], tracker.goals[1], tracker.goals[2])
                
                # this is for logging
                tracker.reward += rew
                
            # check if goal is achieved
            goal_achieved = check_goal(next_state, goal, threshold)
            
            # hindsight action transition
            if goal_achieved:
                replay_buffer[i_level].add((state, action, 0.0, next_state, goal, 0.0, float(done)))
            else:
                replay_buffer[i_level].add((state, action, -1.0, next_state, goal, gamma, float(done)))
                
            # copy for goal transition
            goal_transitions.append([state, action, -1.0, next_state, None, gamma, float(done)])
            
            state = next_state
            
            if done or goal_achieved:
                break
        
        # hindsight goal transition
        # last transition reward and discount is 0
        goal_transitions[-1][2] = 0.0
        goal_transitions[-1][5] = 0.0
        for transition in goal_transitions:
            # last state is goal for all transitions
            transition[4] = next_state
            replay_buffer[i_level].add(tuple(transition))
            
        return next_state, done
        
    
    #   <================ training procedure ================>
    
    for i_episode in range(1, max_episodes+1):
        tracker.reward = 0
        tracker.timestep = 0
        # collecting experience in environment
        state = env.reset()
        last_state, done = run_HAC(k_level-1, state, goal_state, False)
        
        if check_goal(last_state, goal_state, threshold):
          print("################ Solved! ################ ")
          for i in range(k_level):
                HAC[i].save(directory, filename+'_solved_level_{}'.format(i))
        
        # update all levels
        for i in range(k_level):
            HAC[i].update(replay_buffer[i], n_iter, batch_size)
        
        # logging updates:
        log_f.write('{},{}\n'.format(i_episode, tracker.reward))
        log_f.flush()
        
        if i_episode % save_episode == 0:
            for i in range(k_level):
                HAC[i].save(directory, filename+'_level_{}'.format(i))
        
        print("Episode: {}\t Reward: {}".format(i_episode, tracker.reward))
        tracker.reward = 0
    
if __name__ == '__main__':
    train()
 
