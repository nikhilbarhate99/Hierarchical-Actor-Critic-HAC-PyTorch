import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bounds, offset):
        super(Actor, self).__init__()
        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim + state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, action_dim),
                        nn.Tanh()
                        )
        # max value of actions
        self.action_bounds = action_bounds
        self.offset = offset
        
    def forward(self, state, goal):
        return (self.actor(torch.cat([state, goal], 1)) * self.action_bounds) + self.offset
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, H):
        super(Critic, self).__init__()
        # UVFA critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim + action_dim + state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )
        self.H = H
        
    def forward(self, state, action, goal):
        # rewards are in range [-H, 0]
        return -self.critic(torch.cat([state, action, goal], 1)) * self.H
    
class DDPG:
    def __init__(self, state_dim, action_dim, action_bounds, offset, lr, H):
        
        self.actor = Actor(state_dim, action_dim, action_bounds, offset).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim, H).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.mseLoss = torch.nn.MSELoss()
    
    def select_action(self, state, goal):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)
        return self.actor(state, goal).detach().cpu().data.numpy().flatten()
    
    def update(self, buffer, n_iter, batch_size):
        
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action, reward, next_state, goal, gamma, done = buffer.sample(batch_size)
            
            # convert np arrays into tensors
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            goal = torch.FloatTensor(goal).to(device)
            gamma = torch.FloatTensor(gamma).reshape((batch_size,1)).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)
            
            # select next action
            next_action = self.actor(next_state, goal).detach()
            
            # Compute target Q-value:
            target_Q = self.critic(next_state, next_action, goal).detach()
            target_Q = reward + ((1-done) * gamma * target_Q)
            
            # Optimize Critic:
            critic_loss = self.mseLoss(self.critic(state, action, goal), target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Compute actor loss:
            actor_loss = -self.critic(state, self.actor(state, goal), goal).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
                
                
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.critic.state_dict(), '%s/%s_crtic.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location='cpu'))
        self.critic.load_state_dict(torch.load('%s/%s_crtic.pth' % (directory, name), map_location='cpu'))  
        
        
        
        
      
        
        
